#!bin/bash
JSON_FILE=settings/example_1.json

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--json_file)
            JSON_FILE="$2"
            shift 2
            ;;
    esac
done


# Run the script
python main.py --json_file $JSON_FILE

