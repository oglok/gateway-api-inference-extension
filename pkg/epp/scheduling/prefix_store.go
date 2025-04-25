package scheduling

import (
	"context"
	"sync"
	"time"

	"github.com/armon/go-radix"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// PrefixEntry represents a single entry in the prefix store
type PrefixEntry struct {
	PodRef    types.NamespacedName
	LastUsed  time.Time
	ModelName string
}

// PrefixStoreConfig holds configuration for the prefix store
type PrefixStoreConfig struct {
	MaxEntries   int           // Maximum total entries in the store
	MinPrefixLen int           // Minimum prefix length to store
	MaxPrefixLen int           // Maximum prefix length to store
	EntryTTL     time.Duration // Time-to-live for entries
}

// PrefixStore manages prompt prefixes and their pod assignments
type PrefixStore struct {
	tree   *radix.Tree
	mu     sync.RWMutex
	config PrefixStoreConfig
}

// NewPrefixStore creates a new PrefixStore with the given configuration
func NewPrefixStore(config PrefixStoreConfig) *PrefixStore {
	logger := log.FromContext(context.Background())
	logger.V(logging.DEBUG).Info("Creating new PrefixStore", "config", config)
	return &PrefixStore{
		tree:   radix.New(),
		config: config,
	}
}

// AddPrefix adds or updates a prefix entry in the store
func (ps *PrefixStore) AddPrefix(ctx context.Context, prefix string, pod types.NamespacedName, modelName string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	logger := log.FromContext(ctx)

	// Validate prefix length
	if len(prefix) < ps.config.MinPrefixLen {
		return &errutil.Error{
			Code: errutil.BadRequest,
			Msg:  "prefix length is below minimum allowed length",
		}
	}
	if len(prefix) > ps.config.MaxPrefixLen {
		logger.V(logging.DEBUG).Info("Truncating prefix", "originalLength", len(prefix), "maxLength", ps.config.MaxPrefixLen)
		prefix = prefix[:ps.config.MaxPrefixLen]
	}

	// Check if we're updating an existing entry
	if val, exists := ps.tree.Get(prefix); exists {
		entry := val.(*PrefixEntry)
		if entry.PodRef == pod && entry.ModelName == modelName {
			logger.V(logging.DEBUG).Info("Updating existing entry", "prefix", prefix, "pod", pod.String())
			entry.LastUsed = time.Now()
			ps.tree.Insert(prefix, entry)
			return nil
		}
	}

	// Check total entries limit
	if ps.tree.Len() >= ps.config.MaxEntries {
		logger.V(logging.DEBUG).Info("Store at capacity, evicting oldest entry", "currentSize", ps.tree.Len(), "maxSize", ps.config.MaxEntries)
		ps.evictOldest()
	}

	// Add new entry
	entry := &PrefixEntry{
		PodRef:    pod,
		LastUsed:  time.Now(),
		ModelName: modelName,
	}
	ps.tree.Insert(prefix, entry)

	logger.V(logging.DEBUG).Info("Successfully added new prefix entry", "prefix", prefix, "pod", pod.String(), "model", modelName, "totalEntries", ps.tree.Len())
	return nil
}

// FindPodForPrefix finds the best matching pod for a given prefix and model
func (ps *PrefixStore) FindPodForPrefix(ctx context.Context, prefix string, modelName string) (types.NamespacedName, bool) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()

	logger := log.FromContext(ctx)

	if len(prefix) < ps.config.MinPrefixLen {
		logger.V(logging.DEBUG).Info("Prefix too short", "prefix", prefix, "minLength", ps.config.MinPrefixLen)
		return types.NamespacedName{}, false
	}

	if len(prefix) > ps.config.MaxPrefixLen {
		logger.V(logging.DEBUG).Info("Truncating prefix", "originalLength", len(prefix), "maxLength", ps.config.MaxPrefixLen)
		prefix = prefix[:ps.config.MaxPrefixLen]
	}

	// Use LongestPrefix to find the best match
	matchedPrefix, val, found := ps.tree.LongestPrefix(prefix)
	if !found {
		logger.V(logging.DEBUG).Info("No matching prefix found", "prefix", prefix)
		return types.NamespacedName{}, false
	}

	entry := val.(*PrefixEntry)

	// Check if entry has expired or model doesn't match
	if time.Since(entry.LastUsed) > ps.config.EntryTTL {
		return types.NamespacedName{}, false
	}
	if entry.ModelName != modelName {
		return types.NamespacedName{}, false
	}

	// Update LastUsed time for the matched entry
	entry.LastUsed = time.Now()
	ps.tree.Insert(matchedPrefix, entry)

	return entry.PodRef, true
}

// evictOldest removes the oldest entry from the store
func (ps *PrefixStore) evictOldest() {
	var oldestKey string
	var oldestTime time.Time
	first := true

	// Use Walk to find the oldest entry
	ps.tree.Walk(func(key string, value interface{}) bool {
		entry := value.(*PrefixEntry)
		if first || entry.LastUsed.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.LastUsed
			first = false
		}
		return false // continue walking
	})

	if oldestKey != "" {
		ps.tree.Delete(oldestKey)
	}
}

// cleanupExpired removes expired entries
func (ps *PrefixStore) cleanupExpired(ctx context.Context) {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	logger := log.FromContext(ctx)
	now := time.Now()
	var keysToDelete []string

	// Use Walk to find expired entries
	ps.tree.Walk(func(key string, value interface{}) bool {
		entry := value.(*PrefixEntry)
		if now.Sub(entry.LastUsed) > ps.config.EntryTTL {
			keysToDelete = append(keysToDelete, key)
		}
		return false
	})

	// Delete expired entries
	for _, key := range keysToDelete {
		ps.tree.Delete(key)
	}

	if len(keysToDelete) > 0 {
		logger.V(logging.DEBUG).Info("Cleaned up expired entries", "count", len(keysToDelete), "remainingEntries", ps.tree.Len())
	} else {
		logger.V(logging.DEBUG).Info("No expired entries found", "totalEntries", ps.tree.Len())
	}
}

// RunMaintenance performs periodic cleanup of expired entries
func (ps *PrefixStore) RunMaintenance(ctx context.Context) {
	logger := log.FromContext(ctx)
	ticker := time.NewTicker(ps.config.EntryTTL / 2)
	defer ticker.Stop()

	logger.V(logging.DEBUG).Info("Starting maintenance routine", "interval", ps.config.EntryTTL/2)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ps.cleanupExpired(ctx)
		}
	}
}
