
-------------------------------------
installation

ubuntu 22 tout neuf
python is python3

pipenv install

# on installe, on regarde quand ça break
pipenv install jaxlib
pipenv install jax
pipenv install tensorflowjs
pipenv install tensorflow

# on désinstalle, en ayant noté les versions
pipenv uninstall numpy jaxlib jax tensorflowjs tensorflow

# on réinstalle
pipenv install numpy==1.23.5 jaxlib==0.4.4 jax==0.4.4 tensorflow==2.11.0 tensorflowjs==4.2.0

# ça fail
#   leçon: faut tout installer ensemble, et ça va raler sur les compat
#   pour l'instant on va faire sans le jupyterlab

DONE
sudo apt install nvidia-cuda-toolkit
ça fait rien ici


--------------------------

how to run:


    sur ubuntu 22 wsl

    cd /mnt/c/ws/jax-raytracer
    pipenv shell
    python main.py
    python3 -m http.server
    
    
    
    
    