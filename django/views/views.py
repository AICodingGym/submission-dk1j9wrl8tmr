from django.shortcuts import render
from ..models import Category
from ..utils.utils import prefetch_with_limit

def category_list_view(request):
    """
    View to display a list of categories with a limited number of related posts.
    """
    # Fetch all categories
    categories = Category.objects.all()

    # Prefetch a limited number of related posts (e.g., 3 posts per category)
    categories = prefetch_with_limit(categories, 'post_set', limit=3, to_attr='example_posts')

    # Render the template with the categories and their limited posts
    return render(request, 'category_list.html', {'categories': categories})