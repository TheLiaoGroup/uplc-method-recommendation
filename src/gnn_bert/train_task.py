# -*- coding: utf-8 -*-


TASK_CONFIGS = {
    'default': {
        'data_path': None,
        'test_data_path': None,
        'use_stratified_sampling': False,
        'save_dir': '/home/huangzy/uplc/results/default',

        # 基本训练参数
        'model_type': 'gnn',
        'num_epochs': 3000,
        'batch_size': 512,
        'dropout': 0.5,
        'weight_decay': 1e-4,
        'learning_rate': 0.0001,
        'train_ratio': 0.9,
        'criterion':'MSELoss',
        'criterion_beta': 0.5,
        'augment': False,
        'shuffle': True,
        'scheduler': None,

        'seed': 42,

        # GNN相关参数
        'gnn_type': 'GIN',
        'num_layers': 3,
        'hidden_dim': 128,
        'input_dim': None,  # 由数据集自动推断

        # BERT相关参数
        'bert_model_name': 'DeepChem/ChemBERTa-77M-MTR',
        'bert_model_dir': None,  # 可选，预训练模型目录
        'max_seq_length': 128,
        'hidden_dim_bert': 256,  # BERT特定的hidden_dim参数

        # 通用参数
        'model_dir': None,  # 预训练权重路径或模型保存路径
        'finetune': False,
        'gradient_clipping': 1.0,
        'scheduler': None,
        'analyze_data': False,
        'gpu': None,  # 默认自动选择
        'use_physicochemical': True,
    },
    
    'test': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/test',
        'num_epochs': 10},

    '20250801_GNN': {
        'data_path': '/home/huangzy/UPLC/datas/20250801/train/processed_dedup_filtered_add_fg_f-Default-2-90__cleaned_with_labels_k3_train.csv',
        'test_data_path': '/home/huangzy/UPLC/datas/20250801/test/processed_dedup_filtered_add_fg_f-Default-2-90__cleaned_with_labels_k3_test.csv',  # 新增：测试数据集路径，如果提供则不分割训练集
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'gnn_type': 'GIN',
        'batch_size': 512,
        'num_epochs': 3000,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'train_ratio': 0.9,
        'save_dir': '/home/huangzy/UPLC/论文数据/20250801/GNN/Default-2'},

    '20250808_GNN': {
        'data_path': '/home/huangzy/UPLC/datas/20250808/train/Default_Neutral_train.csv',
        'test_data_path': '/home/huangzy/UPLC/datas/20250808/test/Default_Neutral_test.csv',  
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'gnn_type': 'GIN',
        'batch_size': 512,
        'num_epochs': 3000,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'save_dir': '/home/huangzy/UPLC/论文数据/20250808/GNN-Default-Neutral'},


    '20250811_GNN': {
        'data_path': '/home/huangzy/UPLC/datas/20250808/train/Default-2-BLANCE_train.csv',
        'test_data_path': '/home/huangzy/UPLC/datas/20250808/test/Default-2-BLANCE_test.csv',  
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'num_epochs': 3000,
        # 'model_dir': '/home/huangzy/UPLC/论文数据/20250811/GNN-Default-2/best_model.pth',
        'save_dir': '/home/huangzy/UPLC/论文数据/20250811/GNN-Default-2-BLANCE'},

    '20260127_GNN': {
        'data_path': '/home/huangzy/uplc/data/2.train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc/data/2.train_test_split/AM-I-filtered_with_labels_k4_test.csv',  
        'batch_size': 2048,
        'num_epochs': 5000,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.0001,
        'weight_decay': 1e-4,
        'dropout': 0.2, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/BERT_GIN/论文数据/20260127/GNN-AM-I'},

    'GNN_1': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-I-filtered_with_labels_k4_test.csv',  
        'batch_size': 128,
        'num_epochs': 500,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        # 'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.0001,
        'dropout': 0.5, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/GNN-AM-I-new'},
    'GNN_2': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k3_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k3_test.csv',  
        'batch_size': 256,
        'num_epochs': 1000,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'criterion':'MSELoss',
        'criterion_beta': 1,
        # 'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.0001,
        'dropout': 0.5, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/GNN-AM-II-new1-test'},

    'GNN_3': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-III-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-III-filtered_with_labels_k4_test.csv',  
        'batch_size': 256,
        'num_epochs': 5000,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.0001,
        'dropout': 0.2, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/GNN-AM-III-new'},
    
    'test1': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-I-filtered_with_labels_k4_test.csv',  
        'batch_size': 256,
        'num_epochs': 500,
        'num_layers': 2,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.0001,
        'dropout': 0.7, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/GNN-AM-I-new'},
    
    'BERT_1': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-I-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-I-filtered_with_labels_k4_test.csv',  
        'model_type': 'bert',
        'max_seq_length': 128,
        'batch_size': 128,
        'num_epochs': 200,
        'learning_rate': 2e-5,
        'dropout': 0.2,
        # 'scheduler': 'ReduceLROnPlateau',
        'weight_decay': 0.0001,  
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/BERT-AM-I'},
    'BERT_2': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k3_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k3_test.csv',  
        'model_type': 'bert',
        'max_seq_length': 128,
        'batch_size': 128,
        'num_epochs': 200,
        'learning_rate': 1e-5,
        'dropout': 0.2,
        'criterion':'MSELoss',
        'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.0001,  
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/BERT-AM-II'},
    'BERT_3': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-III-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-III-filtered_with_labels_k4_test.csv',  
        'model_type': 'bert',
        'max_seq_length': 128,
        'batch_size': 128,
        'num_epochs': 200,
        'learning_rate': 1e-5,
        'dropout': 0.2,
        'criterion':'MSELoss',
        'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.0001,  
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/BERT-AM-III'},
    'GNN_2_k4': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k4_test.csv',  
        'batch_size': 256,
        'num_epochs': 2000,
        'num_layers': 3,
        'hidden_dim': 32,
        'learning_rate': 0.0005,
        'criterion':'WeightedHuber',
        # 'criterion_beta': 1,
        # 'scheduler': 'CosineAnnealingLR',
        'weight_decay': 0.001,
        'dropout': 0.2, 
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/20260212/GNN-AM-II-k4-3'},

    'BERT_2_k4': {
        'data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k4_train.csv',
        'test_data_path': '/home/huangzy/uplc-method-recommendation/datas/2.train_test_split/AM-II-filtered_with_labels_k4_test.csv',  
        'model_type': 'bert',
        'max_seq_length': 128,
        'batch_size': 128,
        'num_epochs': 200,
        'learning_rate': 1e-4,
        'dropout': 0.5,
        'criterion':'MSELoss',
        'scheduler': 'CosineAnnealingLR',
        'weight_decay': 1e-5,  
        'save_dir': '/home/huangzy/uplc-method-recommendation/results/20260212-BERT-AM-II-k4-1'},
}

