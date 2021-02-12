rm -rf add
nvcc add.cu -o add -I. -std=c++14 \
    --extended-lambda \
    --expt-relaxed-constexpr \
    -gencode=arch=compute_80,code=sm_80 \
    -gencode=arch=compute_86,code=sm_86 \
    -gencode=arch=compute_75,code=sm_75

nsys profile ./add
echo ""
# rm -rf add
