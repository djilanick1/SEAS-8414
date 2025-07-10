#!/bin/bash

PCAP_FILE="236-XLMRat.pcap"
TEMP_STREAM="temp_stream.txt"
FOUND_FILES="dropped_files.txt"

# Clear output file if it exists
> "$FOUND_FILES"

# Get total number of TCP streams
TOTAL_STREAMS=$(tshark -r "$PCAP_FILE" -T fields -e tcp.stream | sort -n | uniq | wc -l)

echo "[*] Total TCP streams found: $TOTAL_STREAMS"
echo "[*] Scanning streams for dropped files..."

# Loop through each stream number
for STREAM_NUM in $(seq 0 $((TOTAL_STREAMS - 1))); do
    # Extract stream content
    tshark -r "$PCAP_FILE" -qz follow,tcp,ascii,$STREAM_NUM > "$TEMP_STREAM"

    # Extract file-looking strings (e.g., .exe, .bat, .dll, etc.)
    grep -oEi '[a-zA-Z0-9_\-\\/:]+\.((exe)|(bat)|(dll)|(txt)|(dat)|(ps1)|(vbs)|(jpg)|(png)|(zip))' "$TEMP_STREAM" >> "$FOUND_FILES"
done

# Deduplicate results
sort -u "$FOUND_FILES" > "${FOUND_FILES}.unique"

echo "[✔] Done. Dropped files found:"
cat "${FOUND_FILES}.unique"
