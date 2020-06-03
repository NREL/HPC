
Clone workshop from github to /scratch  :


clone github.com/gridmod/<workshop> use http

```bash
cd /scratch/$USER
git clone https://github.com/GridMod/MSPCM-Workshop.git
cd MSPCM-Workshop 

```

Make a temporary directory for Plexos:

```bash
mkdir -p /scratch/$USER/tmp
```

Setup license file, EE_reg.xml, referencing HPC License in default location

```bash
rm -f ~/.config/EE_reg.xml
mkdir -p ~/.config/PLEXOS
cat > ~/.config/PLEXOS/EE_reg.xml << EOF
<?xml version="1.0"?>
<XmlRegistryRoot>
  <comms>
    <licServer_IP val="10.60.3.188" />
    <licServer_CommsPort val="399" />
    <LastLicTypeUsed val="server" />
  </comms>
  <UserName val="$USER" />
  <Company val="National Renewable Energy Lab" />
  <UserEmail val="example@example.com" />
  <CompanyCode val="6E-86-2D-7E-E2-FF-E9-1C-21-55-40-A0-45-40-A6-F0" />
</XmlRegistryRoot>
EOF
```


