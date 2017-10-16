# Linelet
A Novel Linelet-based Representation for Line Segment Detection.

Link to the <a href='http://ieeexplore.ieee.org/document/7926451/'>paper</a>.

Followings are brief description of each script.

- Demo_main.m: our method is implemented in this script. For an example demonstration, we set this script to detect from image #22 of the dataset. To run over the whole image, you can modify the loop part (line 31) of the code.

- Demo_evaluation.m: you can check the performance comparison of methods. 

- Demo_evaluation_OriginalYorkUrban.m: same as "Demo_evaluation.m" except the original annotation set [11] is used as ground truth. 

- Directories: "DB" contains the original and proposed annotation sets. We provide a script which downloads the DB. "funcs" and "toolbox" contain several functions, such as steps in the paper and others. "result" store detection results of proposed method. "visualization" stores detection results drawn on each image.

Please note that current version is not well optimized. We plan to optimize and re-write in C-language when it is possible.

If you find the code useful, please consider citing the following

```
@ARTICLE{Namgyu2017TPAMI, 
  author={N. G. Cho and A. Yuille and S. W. Lee}, 
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Novel Linelet-based Representation for Line Segment Detection}, 
  year={2017}, 
  volume={PP}, 
  number={99}, 
  pages={1-1}, 
  doi={10.1109/TPAMI.2017.2703841}, 
  ISSN={0162-8828}, 
  month={},
}
```

