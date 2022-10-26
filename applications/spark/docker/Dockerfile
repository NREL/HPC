# USAGE:
# Do not run this while connected to the VPN.
# docker build --tag spark .

# This container can be converted to a Singularity container on Eagle with these commands:
# Save and upload the docker image to Eagle.
# $ docker save -o spark.tar spark
# $ scp spark_container.tar <username>@eagle.hpc.nrel.gov:/scratch/<username>
# Acquire a compute node.
# $ export SINGULARITY_TMPDIR=/dev/shm
# $ module load singularity-container
# $ singularity build spark.sif docker-archive://spark.tar

# Methodology for operation of this container is based on https://sylabs.io/2018/10/spark-on-singularity/

FROM apache/spark-py

USER root

# Install OpenSSH to communicate between containers, and dropbear SSH server
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y openssh-client dropbear locales \
    && rm -rf /var/lib/apt/lists/*

RUN pip install ipython jupyter pandas pyarrow

# This prevents bash warnings on Eagle.
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Set Dropbear port to 2222 (or whatever port was selected above)
RUN sed -i -e 's@\(DROPBEAR_PORT=\).*@\12222@' /etc/default/dropbear

RUN mkdir /nopt
RUN mkdir /lustre
RUN mkdir /projects
RUN mkdir /scratch

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin
# Required for dropbear
ENV SPARK_SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no"

RUN touch $HOME/.profile \
    && rm -rf $HOME/.cache

CMD [ "bash" ]
