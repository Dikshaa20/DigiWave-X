with open("../corpus/07_data_format.txt", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Step 1: split on blank lines, strip whitespace, drop anything empty
 # hint: text.split("\n\n"), then .strip() each one,
                  # then filter out pieces that are empty after stripping
raw_pieces = [p.strip() for p in text.split("\n\n") if p.strip()]


# Step 2: walk through raw_pieces and apply the heading + short-piece rules
chunks = []
pending_heading = None  # holds a heading piece until we find its paragraph

for piece in raw_pieces:
    is_short = len(piece) < 40          # hint: tune this threshold using your own data
    is_heading = piece.endswith(":")     # hint: is this rule good enough on its own?

    if is_short and is_heading:
        pending_heading = piece
        continue  # don't add it as its own chunk yet

    if is_short and not is_heading:
        continue  # drop it — noise like table fragments

    # by here, piece is a real paragraph
    if pending_heading is not None:
        piece = pending_heading + "\n" + piece
        pending_heading = None

    chunks.append(piece)

# Step 3: any chunk still too long (~1000+ chars) gets split further
# hint: split on ". " to break at sentence boundaries, then regroup
# sentences into sub-chunks that stay under your size limit
final_chunks = []
for chunk in chunks:
    if len(chunk) <= 1000:
        final_chunks.append(chunk)
    else:
        sentences = chunk.split(". ")
        sub_chunk = ""
        for s in sentences:
            if len(sub_chunk) + len(s) < 1000:
                sub_chunk += s + ". "
            else:
                final_chunks.append(sub_chunk.strip())
                sub_chunk = s + ". "
        if sub_chunk.strip():
            final_chunks.append(sub_chunk.strip())
        ...  # your sentence-splitting logic here

print("number of chunks:", len(final_chunks))
for c in final_chunks:
    print("-----", len(c), "chars -----")
    print(c[:200])