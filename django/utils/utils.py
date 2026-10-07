def prefetch_with_limit(queryset, related_field, limit, to_attr):
    """
    Custom function to prefetch a limited number of related objects.

    Args:
        queryset: The base queryset to prefetch related objects for.
        related_field: The related field name to prefetch.
        limit: The number of related objects to fetch.
        to_attr: The attribute name to store the prefetched objects.

    Returns:
        A queryset with the related objects prefetched and attached.
    """
    # Fetch all related objects for the given queryset
    related_objects = (
        queryset.model._meta.get_field(related_field).related_model.objects.filter(
            **{f"{related_field}__in": queryset}
        ).order_by('id')  # Ensure consistent ordering
    )

    # Group related objects by their parent
    grouped_objects = {}
    for obj in related_objects:
        parent_id = getattr(obj, f"{related_field}_id")
        grouped_objects.setdefault(parent_id, []).append(obj)

    # Attach the limited number of objects to each parent
    for parent in queryset:
        setattr(parent, to_attr, grouped_objects.get(parent.id, [])[:limit])

    return queryset

# filepath: /Users/isabellajiang/django-15957/django/urls.py
from django.urls import path
from .views.views import category_list_view

urlpatterns = [
    path('categories/', category_list_view, name='category_list'),
]