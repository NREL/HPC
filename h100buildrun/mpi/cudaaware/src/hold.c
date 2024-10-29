#include <stdio.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
// return true if the file specified by the filename exists
bool file_exists(const char *filename)
{
    FILE *fp = fopen(filename, "r");
    bool is_exist = false;
    if (fp != NULL)
    {
        is_exist = true;
        fclose(fp); // close the file
    }
    return is_exist;
}

void hold(const char *filename, float dt) {
	while ( ! file_exists(filename)) {
		printf("%s\n",filename);
		sleep(dt);
	}
}
/*
int main()
{
    char *filename = "readme.txt";
    hold(filename,1);

    if (file_exists(filename))
        printf("File %s exists", filename);
    else
        printf("File %s doesn't exist.", filename);

    return 0;
}
*/
