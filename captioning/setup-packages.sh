# in a terminal, run the commands
curl -sk https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh

source ~/.bashrc  # or ~/.zshrc.
th -e "print 'I just installed Torch! Yesss.'"

sudo apt-get install build-essential curl git m4 ruby texinfo libbz2-dev libcurl4-openssl-dev libexpat-dev libncurses-dev zlib1g-dev
brew update
brew install luarocks

git clone https://github.com/torch/senna.git
cd senna
wget http://ml.nec-labs.com/senna/senna-v3.0.tgz
tar -xvzf senna-v3.0.tgz
luarocks make rocks/senna-scm-1.rockspec

luarocks install torch7
luarocks install nn
luarocks install nngraph
luarocks install optim
luarocks install cutorch
luarocks install cunn
luarocks install json