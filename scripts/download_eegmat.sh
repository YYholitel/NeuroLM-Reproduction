#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 DEST_DIR" >&2
  exit 2
fi

dest=$1
base=https://physionet.org/files/eegmat/1.0.0
data_base=https://physionet-open.s3.amazonaws.com/eegmat/1.0.0
mkdir -p "$dest"
curl --fail --location --silent --show-error "$base/RECORDS" -o "$dest/RECORDS"
curl --fail --location --silent --show-error "$base/SHA256SUMS.txt" -o "$dest/SHA256SUMS.txt"
curl --fail --location --silent --show-error "$base/README.txt" -o "$dest/README.txt"
curl --fail --location --silent --show-error "$data_base/subject-info.csv" -o "$dest/subject-info.csv"

if [[ $(wc -l < "$dest/RECORDS") -ne 72 ]]; then
  echo "Expected 72 EDF records" >&2
  exit 1
fi

while IFS= read -r filename; do
  if [[ ! $filename =~ ^Subject[0-9]{2}_[12]\.edf$ ]]; then
    echo "Unexpected record name: $filename" >&2
    exit 1
  fi
done < "$dest/RECORDS"

download_one() {
  local filename=$1 expected
  expected=$(awk -v name="$filename" '$2 == name { print $1 }' "$dest/SHA256SUMS.txt")
  if [[ -z $expected ]]; then
    echo "No official checksum for $filename" >&2
    return 1
  fi
  if [[ -f $dest/$filename ]] &&
    printf '%s  %s\n' "$expected" "$dest/$filename" | sha256sum --check --status; then
    return 0
  fi
  curl --fail --location --silent --show-error --retry 5 --max-time 180 \
    "$data_base/$filename" -o "$dest/$filename.part"
  if ! printf '%s  %s\n' "$expected" "$dest/$filename.part" | sha256sum --check --status; then
    echo "Checksum mismatch after download: $filename" >&2
    return 1
  fi
  mv -f "$dest/$filename.part" "$dest/$filename"
}
export -f download_one
export dest data_base
xargs -P 8 -I '{}' bash -c 'download_one "$1"' _ '{}' < "$dest/RECORDS"

(cd "$dest" && sha256sum --check --status SHA256SUMS.txt)
echo "Verified 72 EEGMAT v1.0.0 EDF files against PhysioNet SHA256SUMS.txt"
