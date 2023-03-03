
# Output Folder Structure
data are saved in a structured fashion.   
here I'll show you where the data generated in the previous code snippet will be saved:
```
./data
    |- circles
            |- nm_2
                 |- ds_name 
                         |- npps_256
                                  |- ns_1024
                                          |- test
                                                |- imgs
                                                |- circles_no_10_noise_0.01.mat
                                                |- circles_no_20_noise_0.01.mat
                                                |- circles_no_30_noise_0.01.mat
                                                |- circles_no_40_noise_0.01.mat
                                                |- circles_no_50_noise_0.01.mat
                                              
```
where ```imgs``` contains some images of the randomly sampled models. It has the following structure:
```
imgs
   |- circles_no_10_noise_0.01
                            |- *jpg files
   |- circles_no_20_noise_0.01
                            |- *jpg files 
   |- circles_no_30_noise_0.01
                            |- *jpg files
   |- circles_no_40_noise_0.01
                            |- *jpg files 
   |- circles_no_50_noise_0.01
                            |- *jpg files
```

