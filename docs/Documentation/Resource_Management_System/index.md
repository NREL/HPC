# Lex Allocation Management System
*Lex is a system for managing your high-performance computing (HPC) resource assignments, account information, and tracking utilization.*


## Access

An [NREL HPC account](https://www.nrel.gov/hpc/user-accounts.html) is required to access Lex. To log in to Lex, open a web browser to [https://hpcprojects.nrel.gov](https://hpcprojects.nrel.gov/login/?next=/). Log in with your NREL HPC username and password. An OTP token is not required to authenticate. 

## Requesting an Allocation

The system resource allocation request form is available on Lex. 

Once logged in, the request form buttons will be on the home page. To request an allocation for an out-of-cycle/pilot allocation for use in the current fiscal year, click the current year's button. When the allocation cycle is open, a button will be available for the next fiscal year. 

![image](../../assets/images/Lex/request_buttons.png)

If you have an existing allocation that you need to continue for the next fiscal year and you are one of the leads on it, there is an option to copy the information over to a new request form as a starting point. 

![image](../../assets/images/Lex/copy_to.png)

!!! warning
    An allocation request is required each fiscal year, even if your project is continuing.  

!!! info 
    You should submit one allocation request per funded project. Do not split projects up into multiple allocations.


### Request Type

If your request is for 150,000 AUs or less, check the "Pilot Request" option. Fewer fields are required for pilot requests, so be sure to select this option before filling out the rest of the form. If approved, pilots are granted 150,000 AUs on Kestrel and 10TB of /projects storage by default. Pilot projects can be granted at anytime throughout the year. 

![image](../../assets/images/Lex/request_type.png)

### Project Information

A **project handle** other than the default is required. It is used for directory names and for the account used to submit jobs under the allocation. Years and names are not allowed in the handle, as it will be carried over from year to year if your allocation continues. 
!!! tip
    Use the info buttons next to the fields for more information on the question's requirements.
    ![image](../../assets/images/Lex/tooltip.png)

The **HPC Lead** is the person with primary responsibility for the computational work of the allocation. They are the lead on managing user access to the allocation and permissions for the /projects storage allocation. The **HPC Lead** and **HPC Alternate Lead** are required to have NREL HPC accounts and will be contacted for approving changes to the allocation's user list.

![image](../../assets/images/Lex/contacts.png)

### Computational Resources and Readiness

To calculate the AU request for the allocation, input the software that you will be using and information about the runs that will be using that software. You should add an entry for each major application that you will be running. If there's a software that you will be using but the runs for your allocation won't mainly be using it, or it is insubstantial to the AUs, you don't need to add it. Fractional node values are allowed and should be used if your runs don't require a full node's resources.  You can assign GPU and/or CPU hours to a software. The total AU calculation for all of the software entries is used to automatically populate the resource request for the allocation. 

![image](../../assets/images/Lex/computational_request.png)

Note that awarded AUs can be used on either CPUs or GPUs, but we split them during the request for informational and planning purposes. 

The **Use Pattern** describes how you will use your AUs throughout the year. The closer the pattern matches your actual use, the better priority you will have and the less likely you are to lose unused AUs. 

!!! info "Spread Options"

    **Distribute equally across 4 quarters**: even AU split across quarters. Designed for ongoing projects.

    **Development in Q1, production in Q2 or later**: 10% Q1, 30% in remaining quarters. Designed for projects starting off and need time to develop code and workflow.

    **Start in 2nd Quarter**: 33% in Q2-Q4. Designed for projects with late starts.

    **Use in first half of FY**: 45% in Q1 and Q2, 5% in Q3 and Q4. Designed for projects with mid-year end dates or early milestones.

    **Use in second half of FY**: 5% in Q1 and Q2, and 45% in Q3 and Q4. Designed for projects with mid-year star dates or late milestones.

The **Computational Approach** should be a high-level description of the computational method that the project will use, including what software and what types of calculations you will be doing. 

### Submitting your Request

You can save your request as many times as needed, but once it's submitted you will need to contact HPC Operations to change it. Be sure that you selected the **Pilot Request** option if your request is under 150,000 AUs. 

After you have submitted your project, it will undergo an initital screening and a Technical Readiness Review. You may be contacted by the NREL HPC team with questions, please resond to these emails as soon as possible to ensure your request can be processeed as soon as possible. 

For further information on allocations and how to request one, please visit the [Resource Allocations page.](https://www.nrel.gov/hpc/resource-allocation-requests.html)  

## Managing Users

The allocation PI, Project Lead, and Alternate Project Lead can use Lex to manage the allocation entitlement list. 

To add and remove users, navigate to the "Manage Users" page for the allocation.
![image](../../assets/images/Lex/lex-manage-users-nav.png)

To add users, click the "Add User" button and enter the user's email. Repeat for any additional users. Click "Delete" to remove users from the allocation entitlement list.
**Be sure to click "Submit Changes" to apply your changes.**

![image](../../assets/images/Lex/lex-manage-users.png)


Users who have an existing NREL HPC account will be added to the allocation group and permissions will be granted within 24 hours. 
Otherwise, they will be sent an email invitation to request an account and will be listed in the "Pending Invitations" list until their account has been created. You can cancel an invitation by selecting the "Cancel" button and clicking "Submit Changes". 

## Tracking Allocation Usage

### AU Usage Data 

The sidebar on the Lex homepage shows all of the allocations that you have access to. Select a project to check the resource usage.

![image](../../assets/images/Lex/lex_nav.png)

On the "AU Use Report" page, the "Project Use" section lists all of the systems that the project has an allocation on and corresponding AU usage. The "AUs Charged by User" section provides a breakdown of AU usage by user for the project. By default, the data displayed shows each user's total usage across all systems. To filter this to show data for a specific system, use the system button as shown below. 


![image](../../assets/images/Lex/lex_aus_by_user.png)

#### aus_report Command Line Utility

There is a CLI utility `aus_report` available on NREL HPC systems to track your AU usage. This utility uses the data from Lex to output AU usage information on a per-allocation and per-user basis. Please refer to Lex in the case of an discrepancies. Run `aus_report --help` for more information. 

### Jobs Data

To see the list of jobs run using the allocation, select the "List Jobs" tab. 

![image](../../assets/images/Lex/lex_au_use_report.png)

To filter the job data, such as by a specific user or partition, use the search bar. Click the column headers to sort the list by that feature. 

![image](../../assets/images/Lex/lex-jobs-search.png)


To see a summary about a job, click the job ID.

![image](../../assets/images/Lex/lex-job-details.png)


For any questions or feedback on Lex, please contact [HPC-Help@nrel.gov](mailto:HPC-Help@nrel.gov).
