import pytest
from unittest.mock import Mock
from cartridges.data.resources import BaseStructuredResource


class TestStructuredResource(BaseStructuredResource):
    """Test implementation of BaseStructuredResource"""
    
    def __init__(self, config, test_data):
        self.test_data = test_data
        super().__init__(config)
    
    def _load_data(self):
        return self.test_data


class TestListNestedData:
    """Test cases for the _list_nested_data method"""
    
    def test_simple_dict_leaves_only_true(self):
        """Test simple dictionary with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = {"a": 1, "b": "hello", "c": True}
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("a", "1"), ("b", "hello"), ("c", "True")]
        
        assert sorted(result) == sorted(expected)
    
    def test_simple_dict_leaves_only_false(self):
        """Test simple dictionary with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {"a": 1, "b": "hello"}
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include the dict itself plus leaf values
        assert ("", str(test_data)) in result
        assert ("a", "1") in result
        assert ("b", "hello") in result
        assert len(result) == 3
    
    def test_nested_dict_leaves_only_true(self):
        """Test nested dictionary with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = {
            "user": {
                "name": "John",
                "age": 30,
                "details": {
                    "city": "NYC"
                }
            },
            "status": "active"
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [
            ("user/name", "John"),
            ("user/age", "30"),
            ("user/details/city", "NYC"),
            ("status", "active")
        ]
        
        assert sorted(result) == sorted(expected)
    
    def test_nested_dict_leaves_only_false(self):
        """Test nested dictionary with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {
            "user": {
                "name": "John",
                "age": 30
            }
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include all levels: root dict, nested dict, and leaf values
        paths = [item[0] for item in result]
        assert "" in paths  # root dict
        assert "user" in paths  # nested dict
        assert "user/name" in paths  # leaf value
        assert "user/age" in paths  # leaf value
    
    def test_simple_list_leaves_only_true(self):
        """Test simple list with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = [1, "hello", True]
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("0", "1"), ("1", "hello"), ("2", "True")]
        
        assert sorted(result) == sorted(expected)
    
    def test_simple_list_leaves_only_false(self):
        """Test simple list with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = [1, "hello"]
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include the list itself plus leaf values
        assert ("", str(test_data)) in result
        assert ("0", "1") in result
        assert ("1", "hello") in result
        assert len(result) == 3
    
    def test_nested_list_leaves_only_true(self):
        """Test nested list with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = [1, [2, 3], [4, [5, 6]]]
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [
            ("0", "1"),
            ("1/0", "2"),
            ("1/1", "3"),
            ("2/0", "4"),
            ("2/1/0", "5"),
            ("2/1/1", "6")
        ]
        
        assert sorted(result) == sorted(expected)
    
    def test_mixed_dict_list_leaves_only_true(self):
        """Test mixed dictionary and list structures with leaves_only=True"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = {
            "items": [
                {"name": "item1", "value": 10},
                {"name": "item2", "value": 20}
            ],
            "metadata": {
                "total": 2,
                "tags": ["tag1", "tag2"]
            }
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [
            ("items/0/name", "item1"),
            ("items/0/value", "10"),
            ("items/1/name", "item2"),
            ("items/1/value", "20"),
            ("metadata/total", "2"),
            ("metadata/tags/0", "tag1"),
            ("metadata/tags/1", "tag2")
        ]
        
        assert sorted(result) == sorted(expected)
    
    def test_mixed_dict_list_leaves_only_false(self):
        """Test mixed dictionary and list structures with leaves_only=False"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {
            "items": [{"name": "item1"}],
            "count": 1
        }
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        
        # Should include containers at all levels
        paths = [item[0] for item in result]
        assert "" in paths  # root dict
        assert "items" in paths  # list
        assert "items/0" in paths  # dict in list
        assert "items/0/name" in paths  # leaf value
        assert "count" in paths  # leaf value
    
    def test_primitive_value(self):
        """Test with primitive value (not dict or list)"""
        config = Mock()
        config.leaves_only = True
        config.seed_prompts = []
        
        test_data = "simple string"
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("", "simple string")]
        
        assert result == expected
    
    def test_empty_dict(self):
        """Test with empty dictionary"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = {}
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("", "{}")]
        
        assert result == expected
    
    def test_empty_list(self):
        """Test with empty list"""
        config = Mock()
        config.leaves_only = False
        config.seed_prompts = []
        
        test_data = []
        resource = TestStructuredResource(config, test_data)
        
        result = resource._list_nested_data(test_data)
        expected = [("", "[]")]
        
        assert result == expected