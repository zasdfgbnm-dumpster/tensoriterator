nvcc add.cu -o add -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr -Xptxas -v
nvcc add-subtract.cu -o add-subtract -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr -Xptxas -v
nvcc compare.cu -o compare -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr -Xptxas -v
./add
echo ""
./add-subtract
echo ""
./compare
echo ""
# rm -rf add add-subtract compare
