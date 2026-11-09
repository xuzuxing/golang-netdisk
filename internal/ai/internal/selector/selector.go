// Package selector provides AI provider round-robin selector
// Author: Done-0
// Created: 2025-01-24
package selector

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/Done-0/gin-scaffold/configs"

	"golang.org/x/time/rate"

	rateUtil "github.com/Done-0/gin-scaffold/internal/utils/rate"
)

// ProviderContext represents the context for a selected provider instance
type ProviderContext struct {
	ProviderName string
	InstanceName string
	BaseURL      string
	APIKey       string
	Model        string
	MaxTokens    int
	Temperature  float64
	TopP         float64
	TopK         int
	Timeout      int
	MaxRetries   int
}

// Selector provides round-robin selection of AI providers
type Selector struct {
	instanceCounter uint64
	mu              sync.RWMutex
	keyCounters     map[string]*uint64
	modelCounters   map[string]*uint64
	rateLimiters    map[string]*rate.Limiter
}

// providerInstance represents a provider instance with its name
type providerInstance struct {
	ProviderName string
	Instance     configs.ProviderInstanceConfig
}

// New creates a new selector instance
func New() *Selector {
	return &Selector{
		keyCounters:   make(map[string]*uint64),
		modelCounters: make(map[string]*uint64),
		rateLimiters:  make(map[string]*rate.Limiter),
	}
}

// Select selects the next available provider context, non-blocking, skips rate-limited instances
func (s *Selector) Select(config *configs.AIConfig, requestModel string) (*ProviderContext, error) {
	var instances []providerInstance
	for name, prov := range config.Providers {
		if !prov.Enabled {
			continue
		}
		for _, inst := range prov.Instances {
			if inst.Enabled && len(inst.Keys) > 0 && len(inst.Models) > 0 {
				instances = append(instances, providerInstance{
					ProviderName: name,
					Instance:     inst,
				})
			}
		}
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("no available AI provider")
	}

	startCounter := atomic.LoadUint64(&s.instanceCounter)

	for i := 0; i < len(instances); i++ {
		index := (startCounter + uint64(i)) % uint64(len(instances))
		selected := instances[index]
		counterKey := fmt.Sprintf("%s:%s", selected.ProviderName, selected.Instance.Name)

		var limiter *rate.Limiter
		if selected.Instance.RateLimit != "" {
			s.mu.RLock()
			limiter = s.rateLimiters[counterKey]
			s.mu.RUnlock()

			if limiter == nil {
				s.mu.Lock()
				limiter = s.rateLimiters[counterKey]
				if limiter == nil {
					rateLimit, burst, err := rateUtil.ParseLimit(selected.Instance.RateLimit)
					if err != nil {
						s.mu.Unlock()
						continue
					}
					limiter = rate.NewLimiter(rateLimit, burst)
					s.rateLimiters[counterKey] = limiter
				}
				s.mu.Unlock()
			}

			if !limiter.Allow() {
				continue
			}
		}

		atomic.StoreUint64(&s.instanceCounter, index+1)

		s.mu.Lock()
		keyCounter, exists := s.keyCounters[counterKey]
		if !exists {
			keyCounter = new(uint64)
			s.keyCounters[counterKey] = keyCounter
		}
		s.mu.Unlock()

		keyIndex := atomic.AddUint64(keyCounter, 1) - 1
		apiKey := selected.Instance.Keys[keyIndex%uint64(len(selected.Instance.Keys))]

		var model string
		if requestModel != "" {
			model = requestModel
		} else {
			s.mu.Lock()
			modelCounter, exists := s.modelCounters[counterKey]
			if !exists {
				modelCounter = new(uint64)
				s.modelCounters[counterKey] = modelCounter
			}
			s.mu.Unlock()
			modelIndex := atomic.AddUint64(modelCounter, 1) - 1
			model = selected.Instance.Models[modelIndex%uint64(len(selected.Instance.Models))]
		}

		return &ProviderContext{
			ProviderName: selected.ProviderName,
			InstanceName: selected.Instance.Name,
			BaseURL:      selected.Instance.BaseURL,
			APIKey:       apiKey,
			Model:        model,
			MaxTokens:    selected.Instance.MaxTokens,
			Temperature:  float64(selected.Instance.Temperature),
			TopP:         float64(selected.Instance.TopP),
			TopK:         selected.Instance.TopK,
			Timeout:      selected.Instance.Timeout,
			MaxRetries:   selected.Instance.MaxRetries,
		}, nil
	}

	return nil, fmt.Errorf("all providers are rate limited")
}

// SelectWithWait selects the next provider context, blocking wait, aligns with Go's rateLimiter.Wait
func (s *Selector) SelectWithWait(ctx context.Context, config *configs.AIConfig, requestModel string) (*ProviderContext, error) {
	var instances []providerInstance
	for name, prov := range config.Providers {
		if !prov.Enabled {
			continue
		}
		for _, inst := range prov.Instances {
			if inst.Enabled && len(inst.Keys) > 0 && len(inst.Models) > 0 {
				instances = append(instances, providerInstance{
					ProviderName: name,
					Instance:     inst,
				})
			}
		}
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("no available AI provider")
	}

	index := atomic.AddUint64(&s.instanceCounter, 1) - 1
	selected := instances[index%uint64(len(instances))]
	counterKey := fmt.Sprintf("%s:%s", selected.ProviderName, selected.Instance.Name)

	if selected.Instance.RateLimit != "" {
		s.mu.RLock()
		limiter := s.rateLimiters[counterKey]
		s.mu.RUnlock()

		if limiter == nil {
			s.mu.Lock()
			limiter = s.rateLimiters[counterKey]
			if limiter == nil {
				rateLimit, burst, err := rateUtil.ParseLimit(selected.Instance.RateLimit)
				if err != nil {
					s.mu.Unlock()
					return nil, err
				}
				limiter = rate.NewLimiter(rateLimit, burst)
				s.rateLimiters[counterKey] = limiter
			}
			s.mu.Unlock()
		}

		if err := limiter.Wait(ctx); err != nil {
			return nil, err
		}
	}

	s.mu.Lock()
	keyCounter, exists := s.keyCounters[counterKey]
	if !exists {
		keyCounter = new(uint64)
		s.keyCounters[counterKey] = keyCounter
	}
	s.mu.Unlock()

	keyIndex := atomic.AddUint64(keyCounter, 1) - 1
	apiKey := selected.Instance.Keys[keyIndex%uint64(len(selected.Instance.Keys))]

	var model string
	if requestModel != "" {
		model = requestModel
	} else {
		s.mu.Lock()
		modelCounter, exists := s.modelCounters[counterKey]
		if !exists {
			modelCounter = new(uint64)
			s.modelCounters[counterKey] = modelCounter
		}
		s.mu.Unlock()
		modelIndex := atomic.AddUint64(modelCounter, 1) - 1
		model = selected.Instance.Models[modelIndex%uint64(len(selected.Instance.Models))]
	}

	return &ProviderContext{
		ProviderName: selected.ProviderName,
		InstanceName: selected.Instance.Name,
		BaseURL:      selected.Instance.BaseURL,
		APIKey:       apiKey,
		Model:        model,
		MaxTokens:    selected.Instance.MaxTokens,
		Temperature:  float64(selected.Instance.Temperature),
		TopP:         float64(selected.Instance.TopP),
		TopK:         selected.Instance.TopK,
		Timeout:      selected.Instance.Timeout,
		MaxRetries:   selected.Instance.MaxRetries,
	}, nil
}

// Reset resets all counters and limiters
func (s *Selector) Reset() {
	atomic.StoreUint64(&s.instanceCounter, 0)
	s.mu.Lock()
	s.keyCounters = make(map[string]*uint64)
	s.modelCounters = make(map[string]*uint64)
	s.rateLimiters = make(map[string]*rate.Limiter)
	s.mu.Unlock()
}
