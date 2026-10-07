def prefetch_with_limit(queryset, related_name, limit, to_attr):
    """
    Prefetch a limited number of related objects for each object in the queryset.
    """
    related_model = queryset.model._meta.get_field(related_name).related_model
    limited_queryset = related_model.objects.filter(
        **{f"{queryset.model._meta.model_name}": OuterRef("pk")}
    ).order_by("id")[:limit]
    return queryset.prefetch_related(
        Prefetch(
            related_name,
            queryset=related_model.objects.filter(id__in=Subquery(limited_queryset.values("id"))),
            to_attr=to_attr,
        )
    )

# filepath: /Users/isabellajiang/django-15957/django/urls.py
from django.urls import path
from .views.views import category_list_view

urlpatterns = [
    path('categories/', category_list_view, name='category_list'),
]