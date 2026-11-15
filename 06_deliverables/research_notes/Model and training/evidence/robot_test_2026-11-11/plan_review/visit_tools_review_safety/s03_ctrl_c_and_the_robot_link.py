"""
Safety review of the visit tools, check 3: what does Ctrl+C do when it arrives while the program waits for the robot's
answer (a move under way, the placing aid reading the state)? NOT the robot: an experiment on this computer only.

The robot library (pyniryo 1.2.5) sends a command and then reads one answer: 2 bytes with the length, then that many
bytes of JSON; after connecting it waits for an answer without any time limit (tcp_client.py: settimeout(None)). Here a
stand-in server on 127.0.0.1 answers in that way (a move after 0.8 s, as the robot answers when the move is done), and
the client reads with the library's own receiving code, copied below from pyniryo/api/communication_functions.py
(receive_data, receive_dict, dict_to_packet) and tcp_client.py (the order send, then receive). pyniryo is not imported
and nothing is connected to the robot. A REAL Ctrl+C is sent to this process's own console; for that the script starts
itself once more in a console of its own that is not shown.

What is measured is the link's behaviour on this stand-in; that the robot's own server answers one answer per command
is how the library is built, but it was not measured at the robot.

Run from this folder:
    D:/AI_Project/.venv/Scripts/python.exe s03_ctrl_c_and_the_robot_link.py > s03_ctrl_c_and_the_robot_link.out.txt
"""

import ctypes
import json
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

PACKET_SIZE_INFOS = {"nbr_bytes": 2, "type": "@H"}
READ_SIZE = 512
START = 0.0


# --- copied from pyniryo 1.2.5, api/communication_functions.py (the Python 3 branch) -------------------------------
def receive_data(sckt, packet_size_infos, buffer_size):
    nbr_bytes = packet_size_infos["nbr_bytes"]
    received_data = sckt.recv(nbr_bytes)
    if len(received_data) != nbr_bytes:
        return None
    size_packet = int(struct.unpack(packet_size_infos["type"], received_data)[0])
    received_big_data = b""

    while len(received_big_data) < size_packet:
        received_big_data += sckt.recv(min(buffer_size, size_packet - len(received_big_data)))

    return received_big_data.decode()


def receive_dict(sckt, packet_size_infos = PACKET_SIZE_INFOS, buffer_size = READ_SIZE):
    msg = receive_data(sckt, buffer_size = buffer_size, packet_size_infos = packet_size_infos)
    return None if msg is None else json.loads(msg)


def dict_to_packet(dict_obj, packet_size_infos = PACKET_SIZE_INFOS):
    json_obj = json.dumps(dict_obj).encode()
    return struct.pack(packet_size_infos["type"], len(json_obj)) + json_obj
# ------------------------------------------------------------------------------------------------------------------


def send_n_receive(sckt, command, *parameters):
    """As tcp_client.py does: send the command, then read one answer and take it for this command's."""

    sckt.send(dict_to_packet({"command": command, "param_list": list(parameters)}))
    answer = receive_dict(sckt)
    if not answer:
        raise RuntimeError("HostNotReachableException")
    if answer["status"] != "OK":
        raise RuntimeError(f"Command {answer['command']} KO")
    return answer


def exactly(connection, count):
    """``count`` bytes from ``connection``, or None when it was closed."""

    data = b""
    while len(data) < count:
        part = connection.recv(count - len(data))
        if not part:
            return None
        data += part
    return data


def stand_in_server(listener, received, delays):
    """Serve one client: note every command with its time and answer it after its delay."""

    connection, _ = listener.accept()
    while True:
        header = exactly(connection, 2)
        if header is None:
            return
        command = json.loads(exactly(connection, struct.unpack("@H", header)[0]))
        received.append((round(time.monotonic() - START, 2), command["command"]) + tuple(command["param_list"]))
        time.sleep(delays.get(command["command"], 0.0))
        values = {"GET_POSE": [0.14, -0.015, 0.04, 0.0, 1.5, 0.0], "GET_JOINTS": [-0.1, -0.7, -1.1, 0.0, 0.2, -0.1]}.get(command["command"], [])
        try:
            connection.sendall(dict_to_packet({"status": "OK", "list_ret_param": values, "payload_size": 0, "command": command["command"]}))
        except OSError:
            return


def with_interrupt_kept(call):
    """
    The fix that is proposed: while a call to the robot is under way, Ctrl+C only leaves a mark; KeyboardInterrupt is
    raised once the whole answer has been read, so the link stays in step.
    """

    marks = []
    previous = signal.signal(signal.SIGINT, lambda number, frame: marks.append(number))
    try:
        result = call()
    finally:
        signal.signal(signal.SIGINT, previous)
    if marks:
        raise KeyboardInterrupt
    return result


def experiment(title, first_command, first_delay, ctrl_c_at, follow_up, lines, is_between_calls = False, is_fix_used = False):
    """
    One link, one Ctrl+C: ``first_command`` is asked (its answer comes after ``first_delay`` s), Ctrl+C arrives at
    ``ctrl_c_at`` s, and then ``follow_up`` is asked, which is what the driver does next (the lift reads the state, or
    the motors are switched back on).
    """

    global START
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received = []
    threading.Thread(target = stand_in_server, args = (listener, received, {first_command: first_delay}), daemon = True).start()
    link = socket.create_connection(("127.0.0.1", listener.getsockname()[1]))
    # as the library does after connecting: an answer is waited for without any limit
    link.settimeout(None)
    START = time.monotonic()
    interrupted_at = None
    try:
        if is_between_calls:
            send_n_receive(link, first_command)
        threading.Timer(ctrl_c_at, lambda: ctypes.WinDLL("kernel32").GenerateConsoleCtrlEvent(0, 0)).start()
        if is_between_calls:
            time.sleep(ctrl_c_at + 1.0)
        elif is_fix_used:
            with_interrupt_kept(lambda: send_n_receive(link, first_command))
            time.sleep(1.0)
        else:
            send_n_receive(link, first_command)
            time.sleep(1.0)
    except KeyboardInterrupt:
        interrupted_at = round(time.monotonic() - START, 2)
    # the library would wait without limit; the 4 s here only serve to see that it would
    link.settimeout(4.0)
    asked_at = round(time.monotonic() - START, 2)
    try:
        answer = send_n_receive(link, *follow_up)
        outcome = f"answered, with the answer to {answer['command']}"
    except socket.timeout:
        outcome = "NO ANSWER in 4 s: the driver would wait here for ever"
    except Exception as error:
        outcome = f"failed with {type(error).__name__}: {str(error)[:80]}"
    time.sleep(0.2)
    lines.append(f"== {'WITH THE FIX: ' if is_fix_used else ''}{title}")
    lines.append(f"   Ctrl+C at {ctrl_c_at} s; the answer to {first_command} came after {first_delay} s; KeyboardInterrupt was raised at {interrupted_at} s")
    lines.append(f"   then {' '.join(follow_up)} was asked at {asked_at} s: {outcome}")
    lines.append(f"   what the stand-in server received (seconds, command): {received}")
    link.close()
    listener.close()


def inside_the_hidden_console(result_path):
    """The experiments themselves; the lines go into ``result_path`` (the console is not shown)."""

    lines = []
    try:
        lines.append(f"Ctrl+C switched on for this console: {bool(ctypes.WinDLL('kernel32').SetConsoleCtrlHandler(None, False))}")
        experiment("Ctrl+C while a move is under way, then the lift reads the state", "MOVE", 0.8, 0.3, ("GET_POSE",), lines)
        experiment("Ctrl+C while the placing aid reads the state, then the motors are switched back on", "GET_POSE", 0.4, 0.1, ("SET_LEARNING_MODE", "FALSE"), lines)
        experiment("Ctrl+C between two calls (the program sleeps), then the lift reads the state", "GET_JOINTS", 0.0, 0.3, ("GET_POSE",), lines, is_between_calls = True)
        experiment("Ctrl+C while a move is under way, then the lift reads the state", "MOVE", 0.8, 0.3, ("GET_POSE",), lines, is_fix_used = True)
        experiment("Ctrl+C while the placing aid reads the state, then the motors are switched back on", "GET_POSE", 0.4, 0.1, ("SET_LEARNING_MODE", "FALSE"), lines,
                   is_fix_used = True)
    except BaseException as error:
        lines.append(f"the experiment itself failed: {type(error).__name__}: {error}")
    Path(result_path).write_text("\n".join(lines) + "\n", encoding = "utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inside_the_hidden_console(sys.argv[1])
    else:
        with tempfile.TemporaryDirectory() as folder:
            result_path = Path(folder) / "result.txt"
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = 0
            finished = subprocess.run([sys.executable, __file__, str(result_path)], creationflags = subprocess.CREATE_NEW_CONSOLE, startupinfo = startup, timeout = 240)
            print("a stand-in link on this computer (127.0.0.1), the robot library's receiving code, a real Ctrl+C in a console that is not shown")
            print(f"exit code of the hidden console's script: {finished.returncode}")
            print(result_path.read_text(encoding = "utf-8") if result_path.exists() else "(no result was written)")
