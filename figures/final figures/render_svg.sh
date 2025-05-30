#!/bin/bash
# Exit on error
set -e

input_dir="."
output_dir="png" 
mkdir -p "$output_dir"

if ! command -v convert &> /dev/null
then
    echo "Error: ImageMagick is not installed. Please install it first."
    echo "On Ubuntu/Debian: sudo apt-get install imagemagick"
    echo "On macOS (using Homebrew): brew install imagemagick"
    exit 1
fi

find "$input_dir" -maxdepth 1 -name "*.svg"  -print0 | while IFS= read -r -d $'\0' svg_file; do
  filename=$(basename "$svg_file" .svg)
  png_file="$output_dir/$filename.png"
  convert -background none -density 300 "$svg_file" "$png_file"
  convert -trim "$png_file" "$png_file"
  echo "Converted '$svg_file' to '$png_file'"
done

echo "Conversion completed!"

