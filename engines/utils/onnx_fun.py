from typing import Optional
import torch


class onnx_adds:

    @staticmethod
    def tril_onnx2(inputs: torch.FloatTensor,
                  diagonal: Optional[int] = 0) -> torch.FloatTensor:
        """Caveat to export an tril-based operator with ONNX.
        Args:
            inputs: Input tensor.
            diagonal: Value of diagonal.

        Returns:
            (torch.FloatTensor): Output tensor.
        """
        arange = torch.arange(inputs.size(0), device=inputs.device)
        arange2 = torch.arange(inputs.size(1), device=inputs.device)

        # mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) >= (arange2 - diagonal)
        mask = torch.reshape(arange, (arange.size(0), 1)).expand(-1, inputs.size(1)) >= (arange2 - diagonal)
        return inputs.masked_fill(mask == 0, 0)

    @staticmethod
    def tril_onnx(inputs: torch.FloatTensor,
                  diagonal: Optional[int] = 0) -> torch.FloatTensor:

        inputs_wrapper = []
        if len(inputs.shape) == 2:
            return onnx_adds.tril_onnx2(inputs, diagonal)
        elif len(inputs.shape) == 3:
            for i in range(inputs.size(0)):
                tmp_inputs = torch.clone(inputs[i])
                inputs_wrapper.append(onnx_adds.tril_onnx2(tmp_inputs).tolist())
        elif len(inputs.shape) == 4:
            inputs_wrapper_inner = []
            for i in range(inputs.size(1)):
                tmp_inputs = torch.clone(inputs[0][i])
                inputs_wrapper_inner.append(onnx_adds.tril_onnx2(tmp_inputs).tolist())
            inputs_wrapper.append(inputs_wrapper_inner)
        else:
            raise Exception("not supported inputs shape:" + inputs.shape)

        return torch.tensor(inputs_wrapper, device=inputs.device)


