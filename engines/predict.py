# -*- coding: utf-8 -*-
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import torch
import os
import time


class Predictor:
    def __init__(self, configs, data_manager, device, logger):
        self.device = device
        self.data_manager = data_manager
        self.logger = logger
        self.checkpoints_dir = configs['checkpoints_dir']
        num_labels = len(self.data_manager.categories)
        if configs['model_type'] == 'bp':
            from engines.models.BinaryPointer import BinaryPointer
            self.model = BinaryPointer(num_labels=num_labels).to(device)
        else:
            from engines.models.GlobalPointer import EffiGlobalPointer
            self.model = EffiGlobalPointer(num_labels=num_labels, device=device).to(device)
        self.model.load_state_dict(torch.load(os.path.join(configs['checkpoints_dir'], 'best_model.pkl')))
        self.model.eval()

    def predict_one(self, sentence):
        """
        预测接口
        """
        start_time = time.time()
        encode_results = self.data_manager.tokenizer(sentence, padding='max_length')
        input_ids = encode_results.get('input_ids')
        token_ids = torch.unsqueeze(torch.LongTensor(input_ids), 0).to(self.device)
        attention_mask = torch.unsqueeze(torch.LongTensor(encode_results.get('attention_mask')), 0).to(self.device)
        segment_ids = torch.unsqueeze(torch.LongTensor(encode_results.get('token_type_ids')), 0).to(self.device)
        logits, _ = self.model(token_ids, attention_mask, segment_ids)
        logit = torch.squeeze(logits.to('cpu'))
        results = self.data_manager.extract_entities(sentence, logit)
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        results_dict = {}
        for class_id, result_set in results.items():
            results_dict[self.data_manager.reverse_categories[class_id]] = list(result_set)
        return results_dict

    def predict_test(self):
        pass

    def convert_torch_to_tf(self):
        import onnx
        from onnx_tf.backend import prepare
        max_sequence_length = self.data_manager.max_sequence_length
        dummy_input = torch.ones([1, max_sequence_length]).to('cpu').long()
        dummy_input = (dummy_input, dummy_input, dummy_input)
        onnx_path = self.checkpoints_dir + '/model.onnx'
        torch.onnx.export(self.model.to('cpu'), dummy_input, f=onnx_path, opset_version=10,
                          input_names=['tokens', 'attentions', 'types'], output_names=['logits', 'probs'],
                          do_constant_folding=False,
                          dynamic_axes={'tokens': {0: 'batch_size'}, 'attentions': {0: 'batch_size'},
                                        'types': {0: 'batch_size'}, 'logits': {0: 'batch_size'},
                                        'probs': {0: 'batch_size'}})
        model_onnx = onnx.load(onnx_path)
        tf_rep = prepare(model_onnx)
        pb_path = self.checkpoints_dir + '/model.pb'
        tf_rep.export_graph(pb_path)
        self.logger.info('convert torch to tensorflow pb successful...')
