from rest_framework import serializers
from .models import QA


class AddressesSerializer(serializers.ModelSerializer):
    class Meta:
        model = QA
        fields = ['Q', 'A']

