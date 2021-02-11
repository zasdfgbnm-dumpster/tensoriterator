nvcc add.cu -o add -I. -std=c++14 \
    --extended-lambda \
    --expt-relaxed-constexpr \
    --resource-usage \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_70,code=sm_70

./add
echo ""
# rm -rf add
