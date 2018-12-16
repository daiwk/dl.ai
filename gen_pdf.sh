set -x
set -e

function gen_pdf()
{
    in_file=$1
    out_file=$2
    pandoc -N -s --toc --smart --latex-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in $in_file -o $out_file
}

cd ./c1
gen_pdf ./c1w2.md ../pdfs/c1w2.pdf
gen_pdf ./c1w3.md ../pdfs/c1w3.pdf
gen_pdf ./c1w4.md ../pdfs/c1w4.pdf

cd ../
cd ./c2
gen_pdf ./c2w1.md ../pdfs/c2w1.pdf
gen_pdf ./c2w2.md ../pdfs/c2w2.pdf
gen_pdf ./c2w3.md ../pdfs/c2w3.pdf

cd ../
cd ./c3
gen_pdf ./c3w1.md ../pdfs/c3w1.pdf
gen_pdf ./c3w2.md ../pdfs/c3w2.pdf

cd ../
cd ./c4
gen_pdf ./c4w1.md ../pdfs/c4w1.pdf
gen_pdf ./c4w2.md ../pdfs/c4w2.pdf
gen_pdf ./c4w3.md ../pdfs/c4w3.pdf
gen_pdf ./c4w4.md ../pdfs/c4w4.pdf

cd ../
cd ./c5
gen_pdf ./c5w1.md ../pdfs/c5w1.pdf
gen_pdf ./c5w2.md ../pdfs/c5w2.pdf
gen_pdf ./c5w3.md ../pdfs/c5w3.pdf


