#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

casanovo sequence \
    sample_data/sample_preprocessed_spectra.mgf \
    -c "$SCRIPT_DIR/ting_config.yaml" \
    -m "$SCRIPT_DIR/tiny_model.ckpt" \
    -o test

java -jar "$SCRIPT_DIR/jmzTabValidator.jar" --check inFile=test.mztab | tee val.txt
grep '^\[Error-' val.txt
if [ $? -eq 0 ]; then
    echo "mzTab validation failed:"
    [ -f test.log ] && rm test.log
    [ -f test.mztab ] && rm test.mztab
    [ -f test.mztab ] && rm val.txt
    exit 1
fi

[ -f test.log ] && rm test.log
[ -f test.mztab ] && rm test.mztab
[ -f test.mztab ] && rm val.txt