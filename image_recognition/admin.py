from django.contrib import admin
from .models import img_recog_table
class ad(admin.ModelAdmin):
    list_display=('uploaded_Img','email','processed_image')
admin.site.register(img_recog_table,ad)
# Register your models here.
