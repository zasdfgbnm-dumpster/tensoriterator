echo "Correct:"
nvcc add-subtract.cu -o add-subtract -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./add-subtract
echo "============================="
echo "Bug:"
nvcc add-subtract.cu -D BUG=1 -o add-subtract -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./add-subtract
rm -rf add add-subtract
