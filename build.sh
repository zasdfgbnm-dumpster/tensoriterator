nvcc where.cu -o where -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr
./where
echo ""
rm -rf where
