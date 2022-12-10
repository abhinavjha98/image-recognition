from django.db import models

# Create your models here.
class img_recog_table(models.Model):
    uploaded_Img = models.ImageField(upload_to='images/',null=True,blank=True)
    email=models.CharField(max_length=255) 
    processed_image = models.ImageField(upload_to='images/',null=True,blank=True)