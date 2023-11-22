targethost=Knowcomp-gz
targetfolder=/home/zwanggc/Project/redundancy_is_sparsity/output

rsync -avr -e 'ssh -l zwanggc' $targethost:$targetfolder ./