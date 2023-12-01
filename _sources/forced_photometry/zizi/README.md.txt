# fornax-cloud-access-API
Repository for cloud access considerations


### Server API changes
The suggested changes to the API is to add a column to the SIA (for now, and maybe others later) called `cloud_access`, that has JSON text that describe where the data is located on the cloud.

An example JSON text to access the file `chandra/data/byobsid/0/3480/primary/acisf03480N004_cntr_img2.jpg` from a bucket  called `heasarc-bucket` is given below:

```json
{
    "aws": 
        { 
        "bucket": "heasarc-bucket", 
        "region": "us-east-1", 
        "access": "region", 
        "path": "chandra/data/byobsid/0/3480/primary/acisf03480N004_cntr_img2.jpg" 
        }
}
```
The region of the bucket is given in `"region"`. 

The type of access, which gives information on on data availability, is in `"access"`, and it can be one of: 
- `"region"`: for buckets accessible from withing the same region.
- `"open"`: for buckets that can be accessed from anywhere.
- `"restricted"`: for buckets need some type of authentication.
- `"none"`: for data that are not accessible for some reason.


The following are example XML files for the returned metadata from the [HEASARC](https://heasarc.gsfc.nasa.gov/xamin_aws/vo/sia?table=chanmaster&pos=182.63,39.40&resultformat=text/xml&resultmax=2) and [MAST](https://mast.stsci.edu/portal_vo/Mashup/VoQuery.asmx/SiaV1?MISSION=HST&pos=182.63,39.40) service.


---
