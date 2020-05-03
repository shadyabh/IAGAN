# IAGAN

Image-Adaptive GAN based Reconstruction (https://arxiv.org/pdf/1906.05284.pdf , Accepted to AAAI, 2020).

# Test
1. Place the pre-trained generator at the main directory (for PGGAN we use https://github.com/ptrblck/prog_gans_pytorch_inference)
2. Update "model.py" with the network archeticure.
3. In "main.py" import the generator and update the directory of the pre-trained weights.
4. Define the measurement model.
5. Set the input images directory and the configuration parameters.
6. Run "main.py"

Note:
In "IAGAN.py" we added the bicubic down-sampling and the fft compression models that we used.

# Citation:

    @ARTICLE{IAGAN,
      author = {{Abu Hussein}, Shady and {Tirer}, Tom and {Giryes}, Raja},
      title = "{Image-Adaptive GAN based Reconstruction}",
      journal = {AAAI Conference on Artificial Intelligence},
      year = "2020"
    }
