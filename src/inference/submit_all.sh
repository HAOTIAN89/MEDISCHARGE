# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <integer>"
    exit 1
fi

INTEGER=$1
DIRECTORY_PATH='/home/haotian/make-discharge-me/data/infered'  # set your directory path 

# Construct the file paths
BHC_FILE="${DIRECTORY_PATH}/bhc_7b_infered_${INTEGER}.csv"
DI_FILE="${DIRECTORY_PATH}/di_7b_infered_${INTEGER}.csv"
OUTPUT_FILE="${DIRECTORY_PATH}/submission_${INTEGER}.csv"

python combine_csv.py "$BHC_FILE" "$DI_FILE" "$OUTPUT_FILE"
