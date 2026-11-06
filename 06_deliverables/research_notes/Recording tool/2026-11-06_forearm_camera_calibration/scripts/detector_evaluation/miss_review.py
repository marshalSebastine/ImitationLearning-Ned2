"""
What the pictures without a detection show, labelled by eye (contact sheets and enlarged
crops of every such picture of episodes 037 to 045, final detector, 6.11.26).

Each entry is (episode, first frame, last frame, category, note); the ranges list
only frames the final detector leaves without a detection. Categories:

    no_puck_in_episode       episode 039 was replayed without the puck
    no_puck_placed_yet       start of the episode, the puck not yet put down
    puck_in_hand             the puck is held in a hand (being placed or put back)
    not_in_view              the puck is outside the picture, or entirely hidden
    hidden_in_shadow         the puck is behind the right bracket in the arm's shadow (known from
                             the frames before and after); no red can be seen by eye
    hidden_sliver_visible    the puck is behind the magnet block or a blue bracket of the
                             hand and only a sliver or a cap shows (roughly a third or less;
                             estimated by eye)
    partly_visible_missed    a larger part of the puck is visible but the detector rejects it
"""


MISSED_FRAME_REVIEW = [
    ("037", 46, 46, "hidden_sliver_visible", "sliver at the left bracket"),
    ("037", 86, 90, "hidden_sliver_visible", "dark red patch beside the right bracket, in the arm's shadow"),
    ("037", 91, 91, "not_in_view", "puck hidden behind the hand while the forearm turns"),
    ("037", 92, 92, "hidden_sliver_visible", "dark red patch beside the right bracket, in the arm's shadow"),
    ("037", 145, 145, "partly_visible_missed", "about 40 percent beside the left bracket; rejected as too elongated "
                                               "(axis ratio 3.03, limit 3.0)"),
    ("037", 152, 153, "not_in_view", "puck hidden behind the hand, arm in motion, large shadow"),
    ("037", 154, 154, "hidden_sliver_visible", "sliver at the left bracket"),
    ("037", 171, 171, "hidden_sliver_visible", "pale cap above the block"),
    ("037", 173, 173, "hidden_sliver_visible", "part beside the left bracket (about a third)"),
    ("037", 185, 185, "hidden_sliver_visible", "pale cap above the block"),
    ("037", 187, 187, "not_in_view", "puck hidden behind the hand"),
    ("037", 211, 221, "not_in_view", "camera looks at the cabinets while the tip is lifted and turned; a hand "
                                     "in 214"),
    ("037", 229, 232, "hidden_sliver_visible", "pale cap above the block (a quarter to a third)"),
    ("037", 234, 234, "hidden_sliver_visible", "sliver at the left bracket"),
    ("037", 237, 237, "hidden_sliver_visible", "sliver at the left bracket"),
    ("037", 242, 243, "not_in_view", "puck hidden behind the hand"),
    ("038", 0, 0, "puck_in_hand", "puck held in the fingers above the start"),
    ("038", 12, 13, "hidden_sliver_visible", "sliver between the block and the right bracket"),
    ("038", 15, 16, "hidden_sliver_visible", "sliver between the block and the right bracket"),
    ("038", 17, 17, "not_in_view", "puck behind the right bracket"),
    ("038", 18, 18, "hidden_sliver_visible", "sliver between the block and the right bracket"),
    ("038", 39, 46, "not_in_view", "puck hidden behind the hand"),
    ("038", 47, 47, "hidden_sliver_visible", "sliver at the left bracket"),
    ("038", 51, 51, "hidden_sliver_visible", "sliver at the left bracket"),
    ("038", 52, 62, "not_in_view", "puck hidden behind the hand or left behind"),
    ("041", 0, 0, "puck_in_hand", "puck held in a hand at the top left, being placed"),
    ("041", 112, 114, "hidden_sliver_visible", "dark red patch beside the right bracket, in the arm's shadow "
                                               "(faint in 113 and 114)"),
    ("041", 115, 121, "hidden_in_shadow", "behind the right bracket in the arm's shadow; no red seen by eye"),
    ("041", 122, 122, "puck_in_hand", "a hand at the right holds the puck"),
    ("042", 186, 186, "puck_in_hand", "puck held in a hand during the correction by hand"),
    ("042", 217, 217, "hidden_sliver_visible", "sliver at the left bracket"),
    ("042", 218, 220, "not_in_view", "puck hidden behind the hand"),
    ("043", 17, 53, "not_in_view", "puck left the picture after frame 016; the sheet shows no puck"),
    ("043", 54, 54, "not_in_view", "a hand covers the puck while putting it back (seen from frame 055)"),
    ("043", 90, 127, "not_in_view", "puck outside the picture (seen again far ahead from frame 128)"),
    ("044", 0, 0, "no_puck_placed_yet", "start pose, no puck in the picture (first seen in frame 001)"),
    ("044", 39, 39, "hidden_sliver_visible", "sliver at the right bracket"),
    ("045", 0, 2, "no_puck_placed_yet", "start pose, no puck in the picture (first seen in frame 003)"),
    ("045", 24, 25, "hidden_sliver_visible", "sliver at the right bracket"),
    ("045", 215, 215, "hidden_sliver_visible", "sliver at the right bracket"),
]

# extra pictures of the same camera without a detection
MISSED_EXTRA_PICTURE_REVIEW = {
    "044/camera_check": ("no_puck_placed_yet", "taken at the start pose before frame 000; no puck in the picture"),
    "045/camera_check": ("no_puck_placed_yet", "taken at the start pose before frame 000; no puck in the picture"),
}

# start pictures checked by eye for the puck resting at the start of the line
START_PICTURE_REVIEW = {
    "037/000": "puck resting on the sheet just in front of the block, at the start of the line",
    "039/000": "no puck (episode 039 was replayed without the puck)",
    "044/001": "puck resting in front of the block (first frame with the puck)",
    "045/003": "puck resting in front of the block (first frame with the puck; the arm has moved since frame 000)",
    "038/000": "puck still held in the fingers above the start; not resting",
    "041/000": "puck held in a hand at the top left, being placed; not at the start",
    "042/000": "puck resting on the sheet in front of the block, at the start of the line",
    "043/000": "puck at the start of the line but fingers still touch it",
    "043/camera_check": "puck resting on the sheet in front of the block, at the start of the line",
    "044/000": "no puck in the picture",
    "044/camera_check": "no puck in the picture",
    "045/000": "no puck in the picture (puck first seen in frame 003)",
    "045/camera_check": "no puck in the picture",
}


def review_for_missed_frame(episode: str, frame_index: int) -> tuple:
    """
    Return the by-eye category and note of a frame without a detection.

    Parameters
    ----------
    episode : str
        For example ``"037"``.
    frame_index : int
        The frame's number.

    Returns
    -------
    tuple of str
        ``(category, note)``; ``("unreviewed", "")`` if the frame is not in the review.
    """

    if episode == "039":
        return "no_puck_in_episode", "episode 039 was replayed without the puck"
    for review_episode, first, last, category, note in MISSED_FRAME_REVIEW:
        if review_episode == episode and first <= frame_index <= last:
            return category, note
    return "unreviewed", ""
