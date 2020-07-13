# poladrone_nb

1. Get pytorch Repo into the folder

    ``` sh
    > mkdir ref
    > cd ref
    > git clone https://github.com/Tianxiaomo/pytorch-YOLOv4.git pytorchYOLOv4
    > git checkout 4a5b6c160dad4fa59e6c013ce5e4c9acd40b36d6
    ```

2. Dataset

   - generate a dataset list (.txt) for images paths

3. run training loop

    ``` sh
        python Train.py -b 12 -c 1 -e 3000 \
        -train_txt "...train_list.txt" \
        -val_txt "...val_list.txt"
    ```

   - refer to `train_script`
