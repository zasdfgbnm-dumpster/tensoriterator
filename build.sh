nvcc add.cu -o add -I. -std=c++14 --extended-lambda --expt-relaxed-constexpr --resource-usage
./add
echo ""
# rm -rf add
