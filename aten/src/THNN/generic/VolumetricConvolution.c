#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/VolumetricConvolution.c"
#else

void THNN_(VolumetricConvolution_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          THTensor *weight,
          THTensor *bias,
          THTensor *finput,     // only used by cuda impl
          THTensor *fgradInput, // only used by cuda impl
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THArgCheck(pT != 0 || pW != 0 || pH != 0, 9, "padding not supported by CPU backend");   // sharing signature with CUDA version

  THNN_ARGCHECK(!input->is_empty() && (input->dim() == 4 || input->dim() == 5), 2, input,
		"non-empty 4D or 5D (batch mode) tensor expected for input, but got: %s");

  int dimt = 1;
  int dimh = 2;
  int dimw = 3;

  if (input->dim() == 5)
  {
    dimt++;
    dimh++;
    dimw++;
  }

  int64_t nOutputPlane = weight->size(0);
  int64_t kT           = weight->size(2);
  int64_t kH           = weight->size(3);
  int64_t kW           = weight->size(4);
  int64_t inputDepth   = input->size(dimt);
  int64_t inputHeight  = input->size(dimh);
  int64_t inputWidth   = input->size(dimw);
  int64_t outputDepth  = (inputDepth - kT) / dT + 1;
  int64_t outputWidth  = (inputWidth - kW) / dW + 1;
  int64_t outputHeight = (inputHeight - kH) / dH + 1;
  THTensor *outn = THTensor_(new)();
  int64_t i, j;
  if (input->dim() == 4) /* non-batch mode */
  {
    THTensor_(resize4d)(output, nOutputPlane, outputDepth, outputHeight, outputWidth);

    /* add bias */
    if (bias) {
      for (i = 0; i < THTensor_sizeLegacyNoScalars(bias, 0); i++)
      {
        THTensor_(select)(outn, output, 0, i);
        THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
      }
    } else {
      THTensor_(zero)(output);
    }

    /* do convolutions */
    THTensor_(conv3Dmv)(output, 1.0, 1.0, input, weight, dT, dH, dW, "V", "X");
  }
  else /* batch mode */
  {
    int64_t nBatch = input->size(0);
    THTensor_(resize5d)(output, nBatch, nOutputPlane, outputDepth, outputHeight, outputWidth);
    THTensor *inb = THTensor_(new)();
    THTensor *outb = THTensor_(new)();

    /* loop over batches */
    for (j = 0; j < nBatch; j++)
    {
      THTensor_(select)(inb, input, 0, j);
      THTensor_(select)(outb, output, 0, j);

      /* add bias */
      if (bias) {
        for (i = 0; i < THTensor_sizeLegacyNoScalars(bias, 0); i++)
        {
          THTensor_(select)(outn, outb, 0, i);
          THTensor_(fill)(outn, THTensor_(get1d)(bias, i));
        }
      } else {
        THTensor_(zero)(outb);
      }

      /* do convolutions */
      THTensor_(conv3Dmv)(outb, 1.0, 1.0, inb, weight, dT, dH, dW, "V", "X");
    }

    c10::raw::intrusive_ptr::decref(inb);
    c10::raw::intrusive_ptr::decref(outb);
  }
  c10::raw::intrusive_ptr::decref(outn);
}

void THNN_(VolumetricConvolution_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *weight,
          THTensor *finput, // only used by cuda impl
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH)
{
  THArgCheck(pT != 0 || pW != 0 || pH != 0, 9, "padding not supported by CPU backend");   // sharing signature with CUDA version

  THNN_ARGCHECK(!weight->is_empty() && weight->dim() == 5, 4, weight,
		"non-empty 5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
		"expected for weight, but got: %s");

  int nOutputPlane = (int)THTensor_sizeLegacyNoScalars(weight, 0);

  THNN_ARGCHECK(!gradOutput->is_empty() && (gradOutput->dim() == 4 || gradOutput->dim() == 5), 3,
		gradOutput,
		"non-empty 4D or 5D (batch mode) tensor expected for gradOutput, but got: %s");

  int dimPlane = 0;
  if (gradOutput->dim() == 5)
  {
    dimPlane++;
  }

  THArgCheck(nOutputPlane == gradOutput->size(dimPlane), 1,
    "Number of output features is not equal to nOutputPlane"
  );

  /* gradient to input */
  THTensor *tweight = THTensor_(newTranspose)(weight, 0, 1);
  if (gradOutput->dim() == 4) /* non-batch mode */
  {
    THTensor_(conv3Dmv)(gradInput, 0.0, 1.0, gradOutput, tweight, dT, dH, dW, "F", "C");
  }
  else /* batch mode */
  {
    int64_t nBatch = gradOutput->size(0);
    THTensor *ginpb = THTensor_(new)();
    THTensor *goutb = THTensor_(new)();
    int64_t j;

    THTensor_(resize5d)(gradInput,
      input->size(0), input->size(1), input->size(2), input->size(3), input->size(4)
    );

    /* loop over batches */
    for (j = 0; j < nBatch; j++)
    {
      THTensor_(select)(ginpb, gradInput, 0, j);
      THTensor_(select)(goutb, gradOutput, 0, j);
      THTensor_(conv3Dmv)(ginpb, 0.0, 1.0, goutb, tweight, dT, dH, dW, "F", "C");
    }
    c10::raw::intrusive_ptr::decref(ginpb);
    c10::raw::intrusive_ptr::decref(goutb);
  }

  c10::raw::intrusive_ptr::decref(tweight);
}

void THNN_(VolumetricConvolution_accGradParameters)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THTensor *gradBias,
          THTensor *finput,     // only used by cuda impl
          THTensor *fgradInput, // only used by cuda impl
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          accreal scale_)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(scale_);
  THArgCheck(pT != 0 || pW != 0 || pH != 0, 9, "padding not supported by CPU backend");   // sharing signature with CUDA version

  THNN_ARGCHECK(!gradWeight->is_empty() && gradWeight->dim() == 5, 4, gradWeight,
		"non-empty 5D (nOutputPlane x nInputPlane x kT x kH x kW) tensor "
		"expected for gradWeight, but got: %s");

  int nOutputPlane = (int)THTensor_sizeLegacyNoScalars(gradWeight, 0);
  if (gradBias) {
    THArgCheck(!gradBias->is_empty() && THTensor_nDimensionLegacyNoScalars(gradBias) == 1 && THTensor_sizeLegacyNoScalars(gradBias, 0) == nOutputPlane, 5,
      "gradBias tensor has wrong size"
    );
  }

  int64_t k;
  real *gradBias_data;
  THTensor *gradOutSlice;
  int dimPlane = 0;
  if (gradOutput->dim() == 5)
  {
    dimPlane++;
  }

  THArgCheck(nOutputPlane == gradOutput->size(dimPlane), 1,
    "Number of output features is not equal to nOutputPlane"
  );

  if (gradOutput->dim() == 4) /* non-batch mode */
  {
    /* gradient to bias */
    if (gradBias) {
      gradBias_data = gradBias->data<real>();
      gradOutSlice = THTensor_(new)();
      for (k = 0; k < nOutputPlane; k++)
      {
        THTensor_(select)(gradOutSlice, gradOutput, 0, k);
        gradBias_data[k] += scale * THTensor_(sumall)(gradOutSlice);
      }
      c10::raw::intrusive_ptr::decref(gradOutSlice);
    }

    /* gradient to kernels */
    THTensor_(conv3DRevger)(gradWeight, 1.0, scale, input, gradOutput, dT, dH, dW);
  }
  else /* batch mode */
  {
    int64_t nBatch = gradOutput->size(0);
    THTensor *inpb = THTensor_(new)();
    THTensor *goutb = THTensor_(new)();
    int64_t j;

    /* loop over batches */
    for (j = 0; j < nBatch; j++)
    {
      THTensor_(select)(inpb, input, 0, j);
      THTensor_(select)(goutb, gradOutput, 0, j);

      /* gradient to bias */
      if (gradBias) {
        gradBias_data = gradBias->data<real>();
        gradOutSlice = THTensor_(new)();
        for (k = 0; k < nOutputPlane; k++)
        {
          THTensor_(select)(gradOutSlice, goutb, 0, k);
          gradBias_data[k] += scale * THTensor_(sumall)(gradOutSlice);
        }
        c10::raw::intrusive_ptr::decref(gradOutSlice);
      }

      /* gradient to kernels */
      THTensor_(conv3DRevger)(gradWeight, 1.0, scale, inpb, goutb, dT, dH, dW);
    }
    c10::raw::intrusive_ptr::decref(inpb);
    c10::raw::intrusive_ptr::decref(goutb);
  }
}

#endif
