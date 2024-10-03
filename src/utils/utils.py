


def get_wandb_config(args):
    config={
            'model': args.model,
            'epochs':args.epochs,
            'batch_size':args.batch_size,
            'N':args.N,
            'M':args.M,
            'lambda':args.lmbda,
            'seed': args.seed
    }
    add_config = {}
    
    if(args.model == 'wgrcnn_cw'):
        add_config={
            'knn': args.knn,
            'graph_conv': args.graph_conv,
            'heads': args.local_graph_heads, 
            'use_edge_attr': args.use_edge_attr,
            'dissimilarity': args.dissimilarity 
        }

    return {**config, **add_config}