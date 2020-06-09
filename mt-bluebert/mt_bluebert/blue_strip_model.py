"""
Usage:
    blue_strip_model <src> <dest>
"""
import logging
import os

import docopt
import torch


def main():
    args = docopt.docopt(__doc__)
    print(args)

    if not os.path.exists(args['<src>']):
        logging.error('%s: Cannot find the model', args['<src>'])

    map_location = 'cpu' if not torch.cuda.is_available() else None
    state_dict = torch.load(args['<src>'], map_location=map_location)
    config = state_dict['config']
    if config['ema_opt'] > 0:
        state = state_dict['ema']
    else:
        state = state_dict['state']

    my_state = {k: v for k, v in state.items() if not k.startswith('scoring_list.')}
    my_config = {k: config[k] for k in ('vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
                                        'hidden_act', 'intermediate_size', 'hidden_dropout_prob',
                                        'attention_probs_dropout_prob', 'max_position_embeddings', 'type_vocab_size',
                                        'initializer_range')}

    torch.save({'state': my_state, 'config': my_config}, args['<dest>'])


if __name__ == '__main__':
    main()
