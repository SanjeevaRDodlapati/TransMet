'''
Created on Dec 28, 2020

@author: jsun
'''

def copy_weights(src_model, dst_model):
    """Copy weights from `src_model` to `dst_model`.

    Parameters
    ----------
    src_model
        Keras source model.
    dst_model
        Keras destination model.

    Returns
    -------
    list
        Names of layers that were copied.
    """
    assert len(src_model.layers) == len(dst_model.layers)
        
    copied = dict()    
    layers = zip(src_model.layers, dst_model.layers)
    for src_layer, dst_layer in layers:
        if (len(src_layer.get_weights()) == 0):
            # no weight, skip
            continue
        
        dst_layer.set_weights(src_layer.get_weights())
        copied[src_layer.name] = dst_layer.name
            
    return copied