%nproc=%CPU%
%mem=%MEM%MB
%chk=%CHK%
#P %MTD%/%BSS% scf(maxcycle=200) %OPT%

Gaussian input

%CHG% %SPS%
%CRD%
%MOD%
