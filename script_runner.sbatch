#!/bin/bash                                                                                                                        


#SBATCH --job-name=denis_script                                                                                                
#SBATCH --output=results_experiment.txt                                                                                                  
#SBATCH --error=errors_experiment.err                                                                                                     
#SBATCH --nodes=1                                                                                                                  
#SBATCH --ntasks=1                                                                                                                 
#SBATCH --gres=gpu:1                                                                                                               


#SBATCH --time=48:00:00                                                                                                           
srcdir=/Users/deka6994/tf_notebooks/                                                                                      
wrkdir=/Users/deka6994/tf_notebooks/                                                                                      
export PATH=/Users/deka6994/anaconda2/bin:/Users/deka6994/cuda8/bin:/Users/deka6994/cuda8/include:$PATH                   
export LIBRARY_PATH=:/Users/deka6994/CUDA_lib/lib64:$LD_LIBRARY_PATH                                                      
export LD_LIBRARY_PATH=:/Users/deka6994/CUDA_lib/lib64:/Users/deka6994/CUDA_lib/lib64:$LD_LIBRARY_PATH                    
export CPATH=/Users/deka6994/CUDA_lib/include:$CPATH                                                                      
cd ${wrkdir}                                                                                                              
#source activate snakes                                                                                                            
python attractor_net_notebooks/experiment_script.py