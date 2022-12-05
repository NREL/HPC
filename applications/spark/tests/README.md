# Instructions

1. Change to this directory on Eagle.
2. Run the tests with a valid account.
```
$ ./run_tests.sh ACCOUNT
```

3. The job ID was printed to the console. Wait for it to complete.
4. Verify the results by checking that the exit code is 0. Check the .o and .e files if desired.
```
$ sacct -j JOB_ID
```

5. Delete the `run` directory.
