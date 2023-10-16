# USAGE:
# Do not run this while connected to the VPN.
# docker build --tag spark .

# This container can be converted to an Apptainer container on Eagle with these commands:
# Save and upload the docker image to Eagle.
# $ docker save -o spark.tar spark
# $ scp spark_container.tar <username>@eagle.hpc.nrel.gov:/scratch/<username>
# Acquire a compute node.
# $ export APPTAINER_CACHEDIR=/dev/shm
# $ module load apptainer
# $ apptainer build spark.sif docker-archive://spark.tar

# Methodology for operation of this container is based on https://sylabs.io/2018/10/spark-on-singularity/

FROM apache/spark-r

USER root

# Install OpenSSH to communicate between containers, and dropbear SSH server
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y openssh-client dropbear locales libxml2 libxml2-dev libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN apt-get update
RUN R -e 'install.packages("tidyverse")'
RUN R -e 'install.packages("sparklyr")'
RUN R -e 'install.packages("data.table")'
RUN R -e 'install.packages("here")'
RUN R -e 'install.packages("janitor")'
RUN R -e 'install.packages("skimr")'

# This prevents bash warnings on Eagle.
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Set Dropbear port to 2222 (or whatever port was selected above)
RUN sed -i -e 's@\(DROPBEAR_PORT=\).*@\12222@' /etc/default/dropbear

ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin
# Required for dropbear
ENV SPARK_SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no"

RUN touch $HOME/.profile \
    && rm -rf $HOME/.cache

CMD [ "bash" ]
