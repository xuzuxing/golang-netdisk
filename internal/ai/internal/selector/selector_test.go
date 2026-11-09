// Package selector provides AI provider round-robin selector tests
// Author: Done-0
// Created: 2025-01-24
package selector

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/Done-0/gin-scaffold/configs"
)

func createTestConfig() *configs.AIConfig {
	return &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "instance1",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1", "key2"},
						Models:      []string{"gpt-4", "gpt-3.5-turbo"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  2,
					},
					{
						Name:        "instance2",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key3"},
						Models:      []string{"gpt-4"},
						MaxTokens:   8192,
						Temperature: 0.5,
						TopP:        0.95,
						Timeout:     120,
						MaxRetries:  3,
					},
				},
			},
			"gemini": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "official",
						Enabled:     true,
						BaseURL:     "https://generativelanguage.googleapis.com",
						Keys:        []string{"gemini-key1"},
						Models:      []string{"gemini-pro"},
						MaxTokens:   32768,
						Temperature: 0.4,
						TopP:        0.9,
						TopK:        40,
						Timeout:     180,
						MaxRetries:  1,
					},
				},
			},
			"disabled": {
				Enabled: false,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "disabled-instance",
						Enabled:     true,
						BaseURL:     "https://disabled.api.com",
						Keys:        []string{"disabled-key"},
						Models:      []string{"disabled-model"},
						MaxTokens:   1024,
						Temperature: 0.5,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}
}

func TestSelect_NoProvidersEnabled(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {Enabled: false, Instances: []configs.ProviderInstanceConfig{}},
		},
	}

	result, err := selector.Select(config, "")
	if err == nil {
		t.Error("Expected error when no providers are enabled")
	}
	if result != nil {
		t.Error("Expected nil result when no providers are enabled")
	}
}

func TestSelect_NoInstancesHaveKeys(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "empty",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	result, err := selector.Select(config, "")
	if err == nil {
		t.Error("Expected error when no instances have keys")
	}
	if result != nil {
		t.Error("Expected nil result when no instances have keys")
	}
}

func TestSelect_NoInstancesHaveModels(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "empty",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	result, err := selector.Select(config, "")
	if err == nil {
		t.Error("Expected error when no instances have models")
	}
	if result != nil {
		t.Error("Expected nil result when no instances have models")
	}
}

func TestSelect_FirstEnabledInstance(t *testing.T) {
	selector := New()
	config := createTestConfig()

	result, err := selector.Select(config, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("Expected non-nil result")
	}
	if result.ProviderName != "openai" && result.ProviderName != "gemini" {
		t.Errorf("Expected openai or gemini, got %s", result.ProviderName)
	}
}

func TestSelect_RoundRobinThroughInstances(t *testing.T) {
	selector := New()
	config := createTestConfig()

	results := make([]*ProviderContext, 4)
	for i := 0; i < 4; i++ {
		result, err := selector.Select(config, "")
		if err != nil {
			t.Fatalf("Unexpected error at iteration %d: %v", i, err)
		}
		results[i] = result
	}

	instanceNames := make([]string, 4)
	for i, r := range results {
		instanceNames[i] = r.InstanceName
	}

	if instanceNames[0] == instanceNames[1] && instanceNames[1] == instanceNames[2] {
		t.Error("Expected round-robin through different instances")
	}
}

func TestSelect_RoundRobinThroughKeys(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "multi-key",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1", "key2", "key3"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	keys := make([]string, 4)
	for i := 0; i < 4; i++ {
		result, err := selector.Select(config, "")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		keys[i] = result.APIKey
	}

	if keys[0] != "key1" || keys[1] != "key2" || keys[2] != "key3" || keys[3] != "key1" {
		t.Errorf("Expected key1, key2, key3, key1, got %v", keys)
	}
}

func TestSelect_RoundRobinThroughModels(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "multi-model",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{"gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	models := make([]string, 4)
	for i := 0; i < 4; i++ {
		result, err := selector.Select(config, "")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		models[i] = result.Model
	}

	if models[0] != "gpt-4" || models[1] != "gpt-3.5-turbo" || models[2] != "gpt-4-turbo" || models[3] != "gpt-4" {
		t.Errorf("Expected gpt-4, gpt-3.5-turbo, gpt-4-turbo, gpt-4, got %v", models)
	}
}

func TestSelect_UseRequestModel(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "instance",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{"gpt-4", "gpt-3.5-turbo"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	result1, err := selector.Select(config, "custom-model")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result1.Model != "custom-model" {
		t.Errorf("Expected custom-model, got %s", result1.Model)
	}

	result2, err := selector.Select(config, "another-model")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result2.Model != "another-model" {
		t.Errorf("Expected another-model, got %s", result2.Model)
	}
}

func TestSelect_SkipDisabledProviders(t *testing.T) {
	selector := New()
	config := createTestConfig()

	providerNames := make(map[string]bool)
	for i := 0; i < 10; i++ {
		result, err := selector.Select(config, "")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		providerNames[result.ProviderName] = true
	}

	if providerNames["disabled"] {
		t.Error("Expected to skip disabled provider")
	}
}

func TestSelect_SkipDisabledInstances(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "disabled",
						Enabled:     false,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
					{
						Name:        "enabled",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key2"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	instanceNames := make(map[string]bool)
	for i := 0; i < 5; i++ {
		result, err := selector.Select(config, "")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		instanceNames[result.InstanceName] = true
	}

	if instanceNames["disabled"] {
		t.Error("Expected to skip disabled instance")
	}
	if !instanceNames["enabled"] {
		t.Error("Expected to use enabled instance")
	}
}

func TestSelect_CorrectContextProperties(t *testing.T) {
	selector := New()
	config := createTestConfig()

	result, err := selector.Select(config, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if result.ProviderName == "" {
		t.Error("Expected non-empty ProviderName")
	}
	if result.InstanceName == "" {
		t.Error("Expected non-empty InstanceName")
	}
	if result.BaseURL == "" {
		t.Error("Expected non-empty BaseURL")
	}
	if result.APIKey == "" {
		t.Error("Expected non-empty APIKey")
	}
	if result.Model == "" {
		t.Error("Expected non-empty Model")
	}
	if result.MaxTokens == 0 {
		t.Error("Expected non-zero MaxTokens")
	}
}

func TestReset(t *testing.T) {
	selector := New()
	config := createTestConfig()

	selector.Select(config, "")
	selector.Select(config, "")
	selector.Select(config, "")

	selector.Reset()

	result, err := selector.Select(config, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("Expected non-nil result after reset")
	}
}

func TestSelect_RateLimitSkipsInstance(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "limited",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
						RateLimit:   "1/s",
					},
					{
						Name:        "unlimited",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key2"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	result1, err := selector.Select(config, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result1.InstanceName != "limited" {
		t.Errorf("Expected limited, got %s", result1.InstanceName)
	}

	result2, err := selector.Select(config, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result2.InstanceName != "unlimited" {
		t.Errorf("Expected unlimited (limited should be rate limited), got %s", result2.InstanceName)
	}

	result3, err := selector.Select(config, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result3.InstanceName != "unlimited" {
		t.Errorf("Expected unlimited, got %s", result3.InstanceName)
	}
}

func TestSelect_AllInstancesRateLimited(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "limited1",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
						RateLimit:   "1/s",
					},
					{
						Name:        "limited2",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key2"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
						RateLimit:   "1/s",
					},
				},
			},
		},
	}

	selector.Select(config, "")
	selector.Select(config, "")

	result, err := selector.Select(config, "")
	if err == nil {
		t.Error("Expected error when all instances are rate limited")
	}
	if result != nil {
		t.Error("Expected nil result when all instances are rate limited")
	}
}

func TestSelectWithWait(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "limited",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1"},
						Models:      []string{"gpt-4"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
						RateLimit:   "100/s",
					},
				},
			},
		},
	}

	ctx := context.Background()
	start := time.Now()

	results := make([]*ProviderContext, 3)
	for i := 0; i < 3; i++ {
		result, err := selector.SelectWithWait(ctx, config, "")
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		results[i] = result
	}

	elapsed := time.Since(start)

	for _, result := range results {
		if result == nil {
			t.Error("Expected non-nil result")
		}
	}

	if elapsed > 100*time.Millisecond {
		t.Errorf("Expected to complete in <100ms, took %v", elapsed)
	}
}

func TestSelectWithWait_NoProvidersEnabled(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {Enabled: false, Instances: []configs.ProviderInstanceConfig{}},
		},
	}

	ctx := context.Background()
	result, err := selector.SelectWithWait(ctx, config, "")
	if err == nil {
		t.Error("Expected error when no providers enabled")
	}
	if result != nil {
		t.Error("Expected nil result when no providers enabled")
	}
}

func TestHighConcurrency_10000Requests(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"openai": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "instance1",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key1", "key2"},
						Models:      []string{"gpt-4", "gpt-3.5-turbo"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
					},
					{
						Name:        "instance2",
						Enabled:     true,
						BaseURL:     "https://api.openai.com/v1",
						Keys:        []string{"key3", "key4", "key5"},
						Models:      []string{"gpt-4-turbo"},
						MaxTokens:   8192,
						Temperature: 0.5,
						TopP:        0.95,
						Timeout:     120,
						MaxRetries:  2,
					},
				},
			},
			"gemini": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "official",
						Enabled:     true,
						BaseURL:     "https://generativelanguage.googleapis.com",
						Keys:        []string{"gemini-key1", "gemini-key2"},
						Models:      []string{"gemini-pro", "gemini-pro-vision"},
						MaxTokens:   32768,
						Temperature: 0.4,
						TopP:        0.9,
						TopK:        40,
						Timeout:     180,
						MaxRetries:  1,
					},
				},
			},
		},
	}

	instanceCounts := make(map[string]int)
	keyCounts := make(map[string]int)
	modelCounts := make(map[string]int)
	var mu sync.Mutex

	totalRequests := 10000
	start := time.Now()

	for i := 0; i < totalRequests; i++ {
		result, err := selector.Select(config, "")
		if err != nil {
			t.Fatalf("Unexpected error at iteration %d: %v", i, err)
		}
		if result == nil {
			t.Fatalf("Expected non-nil result at iteration %d", i)
		}

		mu.Lock()
		instanceKey := result.ProviderName + ":" + result.InstanceName
		instanceCounts[instanceKey]++

		keyKey := instanceKey + ":" + result.APIKey
		keyCounts[keyKey]++

		modelKey := instanceKey + ":" + result.Model
		modelCounts[modelKey]++
		mu.Unlock()
	}

	elapsed := time.Since(start)
	t.Logf("10000 requests completed in %v", elapsed)

	for key, count := range instanceCounts {
		if count < 3000 || count > 3700 {
			t.Errorf("Instance %s: expected 3000-3700 requests, got %d", key, count)
		}
	}

	if keyCounts["openai:instance1:key1"] < 1500 || keyCounts["openai:instance1:key1"] > 1800 {
		t.Errorf("Key distribution issue for openai:instance1:key1: %d", keyCounts["openai:instance1:key1"])
	}

	if modelCounts["openai:instance1:gpt-4"] < 1500 || modelCounts["openai:instance1:gpt-4"] > 1800 {
		t.Errorf("Model distribution issue for openai:instance1:gpt-4: %d", modelCounts["openai:instance1:gpt-4"])
	}

	t.Logf("Instance distribution: %v", instanceCounts)
	t.Logf("Key distribution: %v", keyCounts)
	t.Logf("Model distribution: %v", modelCounts)
}

func TestHighConcurrency_ConcurrentSelectWithWait(t *testing.T) {
	selector := New()
	config := &configs.AIConfig{
		Providers: map[string]configs.ProviderConfig{
			"provider1": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "inst1",
						Enabled:     true,
						BaseURL:     "https://api1.example.com",
						Keys:        []string{"k1", "k2", "k3"},
						Models:      []string{"m1", "m2"},
						MaxTokens:   4096,
						Temperature: 0.7,
						TopP:        0.9,
						Timeout:     60,
						MaxRetries:  1,
						RateLimit:   "5000/s",
					},
				},
			},
			"provider2": {
				Enabled: true,
				Instances: []configs.ProviderInstanceConfig{
					{
						Name:        "inst2",
						Enabled:     true,
						BaseURL:     "https://api2.example.com",
						Keys:        []string{"k4", "k5"},
						Models:      []string{"m3"},
						MaxTokens:   8192,
						Temperature: 0.5,
						TopP:        0.95,
						Timeout:     120,
						MaxRetries:  2,
						RateLimit:   "5000/s",
					},
				},
			},
		},
	}

	totalRequests := 10000
	start := time.Now()

	ctx := context.Background()
	var wg sync.WaitGroup
	results := make([]*ProviderContext, totalRequests)

	for i := 0; i < totalRequests; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			result, err := selector.SelectWithWait(ctx, config, "")
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}
			results[index] = result
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)
	t.Logf("10000 concurrent SelectWithWait completed in %v", elapsed)

	instanceCounts := make(map[string]int)
	for _, result := range results {
		if result == nil {
			t.Error("Expected non-nil result")
			continue
		}
		key := result.ProviderName + ":" + result.InstanceName
		instanceCounts[key]++
	}

	for key, count := range instanceCounts {
		if count < 4500 || count > 5500 {
			t.Errorf("Instance %s: expected 4500-5500 requests, got %d", key, count)
		}
	}

	t.Logf("Concurrent distribution: %v", instanceCounts)
}
