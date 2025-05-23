# Generated by Django 5.2.1 on 2025-05-15 18:16

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='SpotifyToken',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.CharField(max_length=100, unique=True)),
                ('access_token', models.CharField(max_length=300)),
                ('refresh_token', models.CharField(blank=True, max_length=300, null=True)),
                ('expires_at', models.DateTimeField()),
                ('scope', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
