from django.db import models



class QA(models.Model):
    Q = models.TextField()
    A = models.TextField()
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created']


