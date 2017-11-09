#!/bin/bash

p2cip=/genoscope2/Data/EGAD00001001991/
p2fastq=/genoscope1/Data/EGAD00001001991_fastq/
p2gz=/genoscope2/Data/EGAD00001001991_gz/

mkdir -p $p2fastq
mkdir -p $p2gz

for fl in $p2cip/*.cip
do 
tname=${fl##*/}
tname=`echo $tname | cut -c2-`
tname=$(echo "$tname" | tr '_' '.')
tname="${tname%%.*}"
if [ -f $p2gz$tname.fastq.gz ]
then
continue
fi

echo ${tname}
#ln -s ${fl} $p2fastq$tname.bam.cip
#java -jar /genoscope2/Code/cip/EGA/EgaDemoClient.jar -pf /genoscope2/Code/cip/LLDeep/.ega -dc $p2fastq$tname.bam.cip -dck xyz
#/software/anaconda2/bin/samtools fastq $p2fastq$tname.bam > $p2fastq$tname.fastq &
#rm $p2gz$tname.bam
pigz $p2fastq$tname.fastq &
done