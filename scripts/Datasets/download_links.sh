
echo "........This will download the tar dataset files............"

wget https://www.openslr.org/resources/12/dev-other.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz
wget https://www.openslr.org/resources/12/test-other.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-other-500.tar.gz

echo "............Download complete..............."
echo "..............now unzipping..............."


tar -xvf test-clean.tar.gz -C test-clean
tar -xvf dev-clean.tar.gz -C dev-clean
tar -xvf train-clean-100.tar.gz -C train-clean-100
tar -xvf dev-other.tar.gz -C dev-other
tar -xvf test-other.tar.gz -C test-other
tar -xvf train-clean-360.tar.gz -C train-clean-360
tar -xvf train-other-500.tar.gz -C train-other-500
