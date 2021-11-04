python test_real.py \
    --K=20\
    --T=1\
    --alpha=0.5\
    --attr_mask=0.15\
    --batch_size=32\
    --beta1=0.01\
    --beta2=0.01\
    --dataset=Pubmed\
    --epochs=200\
    --hidden=64\
    --hop=2\
    --lr=0.01\
    --model=DeGNN\
    --nlayer=3\
    --weight_decay=0.0005\
    --debug

python test_real.py \
    --K=25\
    --T=1\
    --alpha=0.5\
    --attr_mask=0.4\
    --batch_size=32\
    --beta1=0.001\
    --beta2=0.001\
    --dataset=citeseer\
    --epochs=200\
    --hidden=64\
    --hop=2\
    --lr=0.01\
    --model=DeGNN\
    --nlayer=2\
    --weight_decay=0.0005 \
    --debug


python test_real.py \
    --K=30\
    --T=1\
    --alpha=0.2\
    --attr_mask=0.4\
    --batch_size=32\
    --beta1=0.01\
    --beta2=0.01\
    --dataset=Cora\
    --epochs=100\
    --hidden=128\
    --hop=2\
    --lr=0.01\
    --model=DeGNN\
    --nlayer=2\
    --weight_decay=0.0005\
    --debug\
    --init