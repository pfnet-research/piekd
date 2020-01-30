ENV=$1
python train_ensemble_sac.py --env ${ENV} \
    --cpo-select-algo uniform \
    --all-h 256,256,256 \
    --all-d 2,2,2 \
    --all-lr 3e-4,3e-4,3e-4 \
    --cpo-train-sample all
