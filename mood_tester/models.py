# mood_tester/models.py
from django.db import models

class SpotifyToken(models.Model):
    user = models.CharField(max_length=100, unique=True) # Or perhaps a ForeignKey to User
    access_token = models.CharField(max_length=300)
    refresh_token = models.CharField(max_length=300, null=True, blank=True)
    expires_at = models.DateTimeField()
    scope = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Token for {self.user}"